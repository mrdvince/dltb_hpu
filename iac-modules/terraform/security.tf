# Instance security group
resource "aws_security_group" "instance_security_group" {
  name        = "instance-security-group"
  description = "Allow SSH traffic"

  ingress {
    description     = "SSH"
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    cidr_blocks     = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "terraform-instance-security-group"
  }
}