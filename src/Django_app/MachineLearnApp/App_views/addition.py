from datetime import datetime
from django.shortcuts import render
from django.http import HttpResponse


def date_actuelle(request):
    return render(request, 'MachineLearn/date.html', {'date': datetime.now()})

def addition(request, nombre1, nombre2):    
    total = nombre1 + nombre2
    return HttpResponse(total)
    #return render(request, 'MachineLearn/addition.html', locals())

def index(request):
    return render(request, 'MachineLearn/MachineLearn.html')
