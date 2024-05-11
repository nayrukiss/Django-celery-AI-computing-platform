# schedule/views.py
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .models import Schedule
from user.models import User
from .task import process_user_task1, process_user_task2, process_user_task3, process_user_task4, \
    process_user_task5, process_user_task6, process_user_task7, process_user_task8
from oss2 import Auth, Bucket
from django.core.paginator import Paginator
import os



def check_login(fn):
    # 装饰器用于验证登录
    def wrap(request, *args, **kwargs):
        if 'uid' not in request.session or 'kind' not in request.session:
            c_uid = request.COOKIES.get('uid')
            c_kind = request.COOKIES.get('kind')
            if not c_uid or c_kind:
                return HttpResponseRedirect('/user/login')
            else:
                request.session['uid'] = c_uid
                request.session['kind'] = c_kind
        return fn(request, *args, **kwargs)
    return wrap


@check_login
def your_view(request):
    if request.method == 'POST':
        uid = request.COOKIES.get('uid')
        user = get_object_or_404(User, id=uid)
        # parameters = request.POST.get('parameters')
        priority = user.kind
        for i in range(1, 30):  # 从1到1，相当于只循环一次
            parameters = i  # 参数为循环变量i的值
            # 创建任务
            task = Schedule.objects.create(user=user, parameters=parameters, status='queued',
                                           task_type=1, priority=priority)
            queue_name = 'low_priority' if priority == 2 else 'high_priority'
            if queue_name == 'low_priority':
                process_user_task1.apply_async(args=[task.id], queue=queue_name)
            else:
                process_user_task2.apply_async(args=[task.id], queue=queue_name)

        return HttpResponse("任务已提交")

    return render(request, 'schedule/calculate_algorithm.html')


@check_login
def your_view2(request):
    if request.method == 'POST':
        uid = request.COOKIES.get('uid')
        user = get_object_or_404(User, id=uid)
        # data_file = request.FILES['data_file']
        test_text = request.POST['functionCode']
        priority = user.kind
        # file_name = os.path.join('user_images', data_file.name)
        # bucket.put_object(file_name, data_file)
        # image_oss_url = file_name
        image_oss_url = f'user_images\IMDB Dataset.csv'
        # 创建任务
        task = Schedule.objects.create(user=user, ga_function_code=test_text,
                                       status='queued', image_oss_url=image_oss_url, task_type=2, priority=priority)
        queue_name = 'low_priority' if priority == 2 else 'high_priority'
        if queue_name == 'low_priority':
            process_user_task3.apply_async(args=[task.id], queue=queue_name)
        else:
            process_user_task4.apply_async(args=[task.id], queue=queue_name)
        return HttpResponse("任务已提交")
        # 成功提交
    return render(request, 'schedule/calculate_algorithm2.html')


@check_login
def your_view3(request):
    if request.method == 'POST':
        uid = request.COOKIES.get('uid')
        user = get_object_or_404(User, id=uid)
        data_file = request.FILES['data_file']

        def get_parameter_with_default(request, parameter_name, default_value):
            parameter_value = request.POST.get(parameter_name)
            if parameter_value == '':
                return default_value
            return parameter_value

        # 使用示例
        aco_iterations = get_parameter_with_default(request, 'aco_iterations', '1000')
        aco_pheromone = get_parameter_with_default(request, 'aco_pheromone', '1')
        aco_decay = get_parameter_with_default(request, 'aco_decay', '0.03')
        ga_iterations = get_parameter_with_default(request, 'ga_iterations', '1000')
        ga_generation_size = get_parameter_with_default(request, 'ga_generation_size', '600')
        ga_mutation_rate = get_parameter_with_default(request, 'ga_mutation_rate', '0.2')
        saa_iterations = get_parameter_with_default(request, 'saa_iterations', '1000')
        saa_temperature = get_parameter_with_default(request, 'saa_temperature', '1000')
        saa_cooling_rate = get_parameter_with_default(request, 'saa_cooling_rate', '0.999')
        aco_params = [aco_iterations, aco_pheromone, aco_decay]
        ga_params = [ga_iterations, ga_generation_size, ga_mutation_rate]
        saa_params = [saa_iterations, saa_temperature, saa_cooling_rate]
        print(aco_params, ga_params, saa_params)
        priority = user.kind
        file_name = os.path.join('user_images', data_file.name)
        bucket.put_object(file_name, data_file)
        image_oss_url = file_name
        # 创建任务
        task = Schedule.objects.create(user=user, status='queued', image_oss_url=image_oss_url, task_type=3,
                                       priority=priority, process1=0, process2=0, process3=0)
        queue_name = 'low_priority' if priority == 2 else 'high_priority'
        if queue_name == 'low_priority':
            process_user_task5.apply_async(args=[task.id, aco_params, ga_params, saa_params], queue=queue_name)
        else:
            process_user_task6.apply_async(args=[task.id, aco_params, ga_params, saa_params], queue=queue_name)
        return HttpResponse("任务已提交")
        # 成功提交
    return render(request, 'schedule/calculate_algorithm3.html')


@check_login
def user_task(request):
    uid = request.COOKIES.get('uid')
    page_num = request.GET.get('page', 1)
    tasks = Schedule.objects.filter(user_id=uid).all().order_by('id')
    task_type_filter1 = request.GET.get('task_type')
    task_type_filter2 = request.GET.get('status')
    if task_type_filter1:
        tasks = tasks.filter(task_type=task_type_filter1)
    if task_type_filter2:
        tasks = tasks.filter(status=task_type_filter2)
    paginator = Paginator(tasks, 10)
    tasks = paginator.page(page_num)
    get_params = request.GET.copy()
    if 'page' in get_params:
        del get_params['page']
    get_params = get_params.urlencode()
    start_page = max(int(page_num) - 3, 1)
    end_page = min(int(page_num) + 3, paginator.num_pages)
    if end_page == paginator.num_pages - 1:
        end_page -= 1
    return render(request, 'schedule/user_tasks.html', locals())


@check_login
def your_view4(request):
    if request.method == 'POST':
        uid = request.COOKIES.get('uid')
        user = get_object_or_404(User, id=uid)
        data_file = request.FILES['data_file']
        priority = user.kind
        file_name = os.path.join('user_images', data_file.name)
        bucket.put_object(file_name, data_file)
        image_oss_url = file_name
        # image_oss_url = f'user_images\IMDB Dataset.csv'
        # 创建任务
        task = Schedule.objects.create(user=user, status='queued', image_oss_url=image_oss_url, task_type=4, priority=priority)
        queue_name = 'low_priority' if priority == 2 else 'high_priority'
        if queue_name == 'low_priority':
            process_user_task7.apply_async(args=[task.id], queue=queue_name)
        else:
            process_user_task8.apply_async(args=[task.id], queue=queue_name)
        return HttpResponse("任务已提交")
        # 成功提交
    return render(request, 'schedule/calculate_algorithm4.html')


@check_login
def get_status(request, task_id):
    task = Schedule.objects.get(id=task_id)
    status = task.status
    result = task.result
    result2 = task.result2
    result3 = task.result3
    p1_oss_url = task.p1_oss_url
    p2_oss_url = task.p2_oss_url
    p3_oss_url = task.p3_oss_url
    return JsonResponse({
        'status': status,
        'result': result,
        'result2': result2,
        'result3': result3,
        'p1_oss_url': p1_oss_url,
        'p2_oss_url': p2_oss_url,
        'p3_oss_url': p3_oss_url,
    })
