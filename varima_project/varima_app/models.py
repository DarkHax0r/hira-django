from django.db import models

class ParfumData(models.Model):
    date = models.DateField()
    pendapatan = models.FloatField()
    modal = models.FloatField()
    
    class Meta:
        db_table = 'data_parfum'  # ganti dengan nama tabel yang sesuai