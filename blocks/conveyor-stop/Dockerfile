FROM host-gateway:5000/ros2-block-base:humble

RUN pip3 install --no-cache-dir pymodbus

COPY node.py /app/node.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
