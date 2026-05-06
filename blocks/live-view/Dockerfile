FROM host-gateway:5000/ros2-block-base:humble

COPY node.py /app/node.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
