from django.shortcuts import render
from user.models import User
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.


def index_view(request):
    uuid = request.COOKIES.get('uid')
    ukind = request.COOKIES.get('kind')
    s_uuid = request.session.get('uid')
    if ukind == 0:
        user = User.objects.get(id=uuid, user_bool=True)
        return render(request, 'index/index2.html', locals())
    if uuid:
        user = User.objects.get(id=uuid, user_bool=True)
        return render(request, 'index/index.html', locals())
    if s_uuid:
        s_uid = request.session['uid']
        s_kind = request.session['kind']
        user = User.objects.get(id=s_uuid, user_bool=True)
        resp = render(request, 'index/index.html', locals())
        resp.set_cookie('uid', s_uid)
        resp.set_cookie('kind', s_kind)
        return resp
    return HttpResponseRedirect('/user/login')