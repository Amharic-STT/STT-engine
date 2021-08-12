#importing the module 
import logging 

# Create and configure logger 
logging.basicConfig(filename="app.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

# Create an object 
logger=logging.getLogger() 

#Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 


logger.debug("This is just a harmless debug message") 
logger.info("This is just an information for you") 
logger.warning("OOPS!!!Its a Warning")  
logger.critical("The Internet is not working....") 
