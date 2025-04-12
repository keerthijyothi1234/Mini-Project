from django.db import models

# Model to register client details
class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=100)

# Model to store cyberattack predictions
class cyberattack_detection(models.Model):
    Fid = models.IntegerField()
    Protocol = models.CharField(max_length=100)
    Flag = models.CharField(max_length=50)
    Packet = models.IntegerField()
    Source_IP_Address = models.GenericIPAddressField()
    Destination_IP_Address = models.GenericIPAddressField()
    Packet_Size = models.IntegerField()
    Prediction = models.CharField(max_length=100)

# Model to store detection accuracy
class detection_accuracy(models.Model):
    names = models.CharField(max_length=100)
    ratio = models.FloatField()

# Model to store detection ratios
class detection_ratio(models.Model):
    names = models.CharField(max_length=100)
    ratio = models.FloatField()