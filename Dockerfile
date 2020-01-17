FROM python:3.6

RUN apt-get update
RUN apt-get upgrade -y

# Install Python3
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nibabel numpy


# Add the user UID:1000, GID:1000, home at /home
ARG GROUP=user
ARG USER=user
ARG UID=1000
ARG GID=1000

RUN groupadd -f -r $GROUP -g $GID && useradd -u $UID -r -g $GID -m -d /home -s /sbin/nologin -c "User" $USER && chmod 755 /home
ENTRYPOINT [ "/bin/bash", "-c" ]
# Set the working directory to app home directory
WORKDIR /home
COPY *.py /home/

# Specify the user to execute all commands below
USER $USER
