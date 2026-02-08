# Frontend Dockerfile - Static files served by nginx
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy static frontend files
COPY index.html /usr/share/nginx/html/
COPY main.js /usr/share/nginx/html/
COPY monitoring.html /usr/share/nginx/html/
COPY monitoring.js /usr/share/nginx/html/
COPY components/ /usr/share/nginx/html/components/

# Production config.js is generated at runtime via entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
