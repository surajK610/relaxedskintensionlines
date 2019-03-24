server {

    server_name langer.orstencil.com www.langer.orstencil.com;

    location / {
        proxy_set_header   X-Forwarded-For $remote_addr;
        proxy_set_header   Host $http_host;
        proxy_pass         "http://127.0.0.1:8000";
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";

    }
 # managed by Certbot

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/langer.orstencil.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/langer.orstencil.com/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot


}server {
    if ($host = www.langer.orstencil.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    if ($host = langer.orstencil.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot



    server_name langer.orstencil.com www.langer.orstencil.com;
    listen 80;
    return 404; # managed by Certbot




}