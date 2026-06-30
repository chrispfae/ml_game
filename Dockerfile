# Wir nutzen ein minimalistisches Nginx-Image
FROM nginx:alpine

# Kopiere deine Dateien in das Standard-Verzeichnis von Nginx
COPY . /usr/share/nginx/html/

# Exponiere Port 80
EXPOSE 80