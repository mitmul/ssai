#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
              "The backend {leveldb, lmdb} containing the images");

std::vector<double> compute_image_mean(
  const std::string &db_file,
  const std::string &save_file) {

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(db_file, db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  BlobProto sum_blob;
  int count = 0;
  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());

  if (DecodeDatum(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    DecodeDatum(&datum);

    const std::string &data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
                                  datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
                                       size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
                          static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << save_file;
  WriteProtoToBinaryFile(sum_blob, save_file);

  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<double> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;

  std::vector<double> means;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
    means.push_back(mean_values[c] / dim);
  }

  return means;
}

void compute_image_stddev(
  const std::vector<double> &means,
  const std::string &db_file) {

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(db_file, db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());

  if (DecodeDatum(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  std::vector<double> stddev_values;
  for (int c = 0; c < datum.channels(); ++c) {
    stddev_values.push_back(0.0);
  }

  int files = 0;
  unsigned long count = 0;
  LOG(INFO) << "Starting Iteration";
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    DecodeDatum(&datum);

    const int channels = datum.channels();
    const int height = datum.height();
    const int width = datum.width();
    const std::string &data = datum.data();
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          const int index = c * height * width + h * width + w;
          const int pixel = static_cast<uint8_t>(data[index]);
          stddev_values[c] += pow((double)pixel - means[c], 2.0);
        }
      }
    }

    count += width * height;
    ++files;

    if (files % 10000 == 0) {
      LOG(INFO) << "Processed " << files << " files.";
      LOG(INFO) << "count: " << count;
      for (int c = 0; c < channels; ++c) {
        LOG(INFO) << "stddev: " << stddev_values[c];
      }
    }

    cursor->Next();
  }
  if (files % 10000 != 0) {
    LOG(INFO) << "Processed " << files << " files.";
    LOG(INFO) << "count: " << count;
  }
  LOG(INFO) << "Finished Iteration";

  std::cout.precision(15);
  LOG(INFO) << "Number of channels: " << datum.channels();
  for (int c = 0; c < datum.channels(); ++c) {
    stddev_values[c] /= (double)count;
    stddev_values[c] = sqrt(stddev_values[c]);
    LOG(INFO) << "stddev_value channel [" << c << "]:"
              << std::fixed << stddev_values[c];
  }
}

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
                          " a leveldb/lmdb\n"
                          "Usage:\n"
                          "    compute_image_mean [FLAGS] INPUT_DB "
                          " MEAN_OUTPUT_FILE STDDEV_OUTPUT_FILE\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  std::vector<double> means = compute_image_mean(argv[1], argv[2]);
  compute_image_stddev(means, argv[1]);

  return 0;
}

