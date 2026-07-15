# Stage 1 — build the React app
FROM node:20-slim AS builder

WORKDIR /app

# Accept the API URL at build time (Vite bakes it into the bundle)
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY package.json .
RUN npm install

COPY . .
RUN npm run build

# Stage 2 — serve the built files with nginx
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config: serve index.html for all routes (SPA support)
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
