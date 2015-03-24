import json
import subprocess
from multiprocessing import Process

keyname = '/home/ubuntu/.ssh/aolab-saito-tokyo.pem'

exe = subprocess.check_output
cmd = 'aws ec2 describe-instances '
cmd += '--filters Name=instance-state-name,Values=running '
cmd += 'Name=instance-type,Values=g2.2xlarge'
instances = exe(cmd, shell=True)
instances = json.loads(instances)['Reservations']
ips = []
for instance in instances:
    instance = instance['Instances'][0]
    ip = instance['NetworkInterfaces'][0]['Association']['PublicIp']
    ips.append(ip)


def do_parallel(func, ips):
    workers = [Process(target=func, args=(ip,)) for ip in ips]
    map(lambda w: w.start(), workers)
    map(lambda w: w.join(), workers)


def exec_on(ip, cmd):
    PATH = 'PATH=/home/ubuntu/anaconda/bin:'
    PATH += '/usr/local/cuda/bin:'
    PATH += '/usr/local/sbin:/usr/local/bin:'
    PATH += '/usr/sbin:/usr/bin:'
    PATH += '/sbin:/bin:/usr/games:/usr/local/games;'
    PYTHONPATH = 'PYTHONPATH=/home/ubuntu/Libraries/caffe/python:'
    PYTHONPATH += '/home/ubuntu/Codes/mapgen/script/lib;'
    LD_LIBRARY_PATH = 'LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:'
    LD_LIBRARY_PATH += '/usr/local/cuda/lib64;'
    cmd_ = ['ssh', '-o', 'StrictHostKeyChecking=no',
            '-i', '/Users/saito/.ssh/aolab-saito-tokyp.pem',
            'ubuntu@%s' % ip,
            'export', PATH,
            'export', PYTHONPATH,
            'export', LD_LIBRARY_PATH]
    cmd_ += cmd
    res = exe(cmd_)
    print res
    return res


def get(ip):
    print exe(
        ['rsync', '-avz', '-e',
         'ssh -i %s -o StrictHostKeyChecking=no' % keyname,
         'ubuntu@%s:/home/ubuntu/Codes/ssai/models' % ip,
         '%s' % ip])

do_parallel(get, ips)
