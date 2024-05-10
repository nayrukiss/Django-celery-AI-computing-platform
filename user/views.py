from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages
from .models import User
import hashlib
from cryptography.hazmat.backends import default_backend
# Create your views here.


def reg_view(request):
    #   register
    if request.method == 'GET':
        return render(request, 'user/register.html')
    #   data
    elif request.method == 'POST':
        username = request.POST['username']
        name = request.POST['name']
        tel = request.POST['tel']
        email = request.POST['email']
        password_1 = request.POST['password_1']
        password_2 = request.POST['password_2']
        if not (username or name or tel or email or password_1 or password_2):
            messages.add_message(request, messages.SUCCESS, '数据输入不能为空!')
            return HttpResponseRedirect('/user/reg')
        #   test1-password
        if password_1 != password_2:
            messages.add_message(request, messages.SUCCESS, '两次输入不一致！')
            return HttpResponseRedirect('/user/reg')
        #   test2-username
        old_users = User.objects.filter(username=username, user_bool=True)
        if old_users:
            messages.add_message(request, messages.SUCCESS, '该工号已经注册')
            return HttpResponseRedirect('/user/reg')
        #   hash
        m = hashlib.md5()
        m.update(password_1.encode())
        password_m = m.hexdigest()
        #   insert
        try:
            user = User.objects.create(username=username, name=name, tel_number=tel, email=email, password=password_m)
        except Exception as e:
            print('--create user %s'%(e))
            messages.add_message(request, messages.SUCCESS, '该工号已经注册')
            return HttpResponseRedirect('/user/reg')
        # request.session['username'] = username
        request.session['uid'] = user.id
        request.session['kind'] = user.kind
        if user.kind == 1:
            resp = HttpResponseRedirect('/index')
        else:
            resp = HttpResponseRedirect('/index')
        resp.set_cookie('uid', user.id, 3600 * 24 * 3)
        resp.set_cookie('kind', user.kind, 3600 * 24 * 3)
        return resp


def login_view(request):

    if request.method == 'GET':
        #   session
        if request.session.get('uid'):
            if request.session.get('kind') == 1:
                return HttpResponseRedirect('/index')
            else:
                return HttpResponseRedirect('/index')
        #   cookies
        cookies_uid = request.COOKIES.get('uid')
        cookies_kind = request.COOKIES.get('kind')
        if cookies_uid:
            request.session['uid'] = cookies_uid
            request.session['kind'] = cookies_kind
            if cookies_kind == 1:
                return HttpResponseRedirect('/index')
            else:
                return HttpResponseRedirect('/index')
        return render(request, 'user/login.html')

    elif request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            user = User.objects.get(username=username)
        except Exception as e:
            print('--login user error %s'%(e))
            messages.add_message(request, messages.SUCCESS, '工号或密码错误！')
            return HttpResponseRedirect('/user/login')
        #   hash
        m = hashlib.md5()
        m.update(password.encode())
        if m.hexdigest() != user.password:
            messages.add_message(request, messages.SUCCESS, '工号或密码错误！')
            return HttpResponseRedirect('/user/login')
        #   record
        # request.session['username'] = username
        request.session['uid'] = user.id
        request.session['kind'] = user.kind
        if user.kind == 1:
            resp = HttpResponseRedirect('/index')
        else:
            resp = HttpResponseRedirect('/index')
        resp.set_cookie('uid', user.id, 3600 * 24 * 3)
        resp.set_cookie('kind', user.kind, 3600 * 24 * 3)
        #   rem
        return resp


def logout_view(request):
    if 'uid' in request.session:
        del request.session['uid']
    if 'kind' in request.session:
        resp = HttpResponseRedirect('/user/login')
        del request.session['kind']
    if 'uid' in request.COOKIES:
        resp.delete_cookie('uid')
    if 'kind' in request.COOKIES:
        resp.delete_cookie('kind')
    return resp
