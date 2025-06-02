import logging 
import other_module
# 确保需要展示的logging level，默认从warning开始,注意format配置在所有logging之前
logging.basicConfig(level=logging.DEBUG,format='%(levelname)s:%(asctime)s:%(message)s (Line %(lineno)d)[%(filename)s]',
                    datefmt='%H:%M:%S',filename='logfile.log')

x : int = 10+10
logging.info('the value of x is %s', x)
logging.info(f"the value of x is {x}")

# 使用logging的format可以对格式进行调整
logging.info('hello world')
logging.debug('DEBUG')

other_module.func()