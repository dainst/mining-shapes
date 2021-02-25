# Mininig Pages WebApp

Web application to apply mining pages pipeline to pages of scanned shapes

## Initialization of DB
```
cd MP_WebApp     
python manage.py migrate
```

## Start App
```
cd MP_WebApp 
/redis-5.0.8/src/redis-server /etc/redis/6379.conf
celery -A MP_WebApp worker --loglevel=info    
python manage.py runserver  #in different terminal window
```