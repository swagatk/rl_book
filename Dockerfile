FROM tensorflow/tensorflow:2.18.0-gpu
WORKDIR /app/rl_book

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \ 
    pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY  * ./

# update apt repoisitories and install sudo
RUN apt-get update && apt-get install sudo
RUN useradd swg 
RUN usermod -aG sudo app
RUN echo "app ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER swg

CMD ["python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))'"]