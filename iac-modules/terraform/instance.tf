resource "aws_key_pair" "habana" {
  key_name   = "habana"
  public_key = file(var.PUBLIC_KEY)
}

resource "aws_instance" "habana" {
  key_name      = aws_key_pair.habana.key_name
  ami           = "ami-092e7d01d03078ccf" # Deep Learning AMI Habana TensorFlow 2.5.0 SynapseAI 0.15.4 (Ubuntu 18.04) 20220105 
  instance_type = "dl1.24xlarge"

  user_data = file(var.USER_DATA)

  tags = {
    Name = "habana"
  }

  vpc_security_group_ids = [
    aws_security_group.instance_security_group.id
  ]

  connection {
    type        = "ssh"
    user        = "habana"
    private_key = file("key")
    host        = self.public_ip
  }

  ebs_block_device {
    device_name = "/dev/sda1"
    volume_type = "gp2"
    volume_size = 500
  }
}

resource "aws_eip" "habana" {
  vpc      = true
  instance = aws_instance.habana.id
}

output "eip" {
  value = aws_eip.habana.public_ip
}
