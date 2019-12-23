FROM centos:7

RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm && yum update -y && yum upgrade -y
# this step cannot be combined with the previous one!
RUN yum install -y python36u
RUN pip3 install nibabel numpy

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
