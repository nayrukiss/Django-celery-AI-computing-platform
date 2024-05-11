# tasks.py
# celery -A schedule worker --loglevel=info -Q high_priority,low_priority -n worker1@%h --autoscale=4,6 -Ofair -E
# celery -A schedule worker --loglevel=info -c 2  -Q high_priority -n worker1@%h -E
# celery -A schedule worker --loglevel=info -c 0  -Q low_priority -n worker2@%h -E
# celery -A schedule worker --loglevel=info -c 4  -Q high_priority,low_priority -n worker3@%h -Ofair -E
# celery -A schedule worker --loglevel=info -c 2 -Q monitor -n worker3@%h -E
# celery -A schedule flower
# celery -A schedule beat -l info
# http://localhost:5555/
import csv
from celery import shared_task, group, chord, chain, current_app
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from schedule.models import Schedule
from time import sleep
import os
import multiprocessing
import random
from oss2 import Auth, Bucket
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from io import StringIO
import networkx as nx
import acopy
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from simanneal import Annealer
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from io import BytesIO
from PIL import Image
import base64
import json
import redis
from tqdm import tqdm
logger = get_task_logger(__name__)
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# 设置 OSS 访问凭证等信息



@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task1(self, task_id):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        random_sleep_time = random.uniform(10, 30)
        sleep(random_sleep_time)
        # 结果
        result = task.parameters
        # 完成
        task.status = 'completed'
        task.result = result
        task.save()
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task2(self, task_id):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        random_sleep_time = random.uniform(10, 30)
        sleep(random_sleep_time)
        # 结果
        result = task.parameters
        # 完成
        task.status = 'completed'
        task.result = result
        task.save()
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task3(self, task_id):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        test_text = task.ga_function_code
        file_name = task.image_oss_url
        image_file = bucket.get_object(file_name)
        data = pd.read_csv(image_file)
        # 朴素贝叶斯分类器
        X = data['review']
        y = data['sentiment']
        # 装换为词频向量
        vector = CountVectorizer()
        # 转化为特征矩阵
        X = vector.fit_transform(X)
        # 多项式分类器
        clf = MultinomialNB()
        # 特征矩阵和情感标签
        clf.fit(X, y)
        # 待测文本处理为词频向量
        test_vector = vector.transform([test_text])
        prediction = clf.predict(test_vector)[0]
        result = prediction
        # 完成
        task.status = 'completed'
        task.result = result
        task.save()
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


@shared_task
def process_user_task4(task_id):
    task = Schedule.objects.get(id=task_id)
    # 进程信息
    task.process_num = os.getpid()
    task.process_info = multiprocessing.current_process().name
    task.save()
    # 运行
    task.status = 'running'
    task.save()
    test_text = task.ga_function_code
    file_name = task.image_oss_url
    image_file = bucket.get_object(file_name)
    data = pd.read_csv(image_file)
    X = data['review']
    y = data['sentiment']
    vector = CountVectorizer()
    X = vector.fit_transform(X)
    clf = MultinomialNB()
    clf.fit(X, y)
    test_vector = vector.transform([test_text])
    prediction = clf.predict(test_vector)[0]
    result = prediction
    # 完成
    task.status = 'completed'
    task.result = result
    task.save()


def distance(coord1, coord2):
    vector1 = np.array(coord1)
    vector2 = np.array(coord2)
    return np.linalg.norm(vector1-vector2)


def create_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = distance(coordinates[i], coordinates[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix


def plot_path(cities, coordinates, path, task_id):
    plt.figure(figsize=(8, 6))
    plt.scatter(np.array(coordinates)[:, 0], np.array(coordinates)[:, 1], c='blue', marker='o')
    for i, c in enumerate(cities):
        plt.text(coordinates[i][0], coordinates[i][1], c, fontsize=9)
    for edge in path:
        start_city, end_city = edge
        start_index = cities.index(start_city)
        end_index = cities.index(end_city)
        start_coord = coordinates[start_index]
        end_coord = coordinates[end_index]
        plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], c='red', linewidth=1)
    plt.title('ACO Best Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    temp_file_path = f'{task_id}_plot.png'
    plt.savefig(temp_file_path)
    plt.close()
    with open(temp_file_path, 'rb') as file:
        bucket.put_object(f'plots/{task_id}_plot1.png', file)
    os.remove(temp_file_path)
    task = Schedule.objects.get(id=task_id)
    task.p1_oss_url = f'https://hxkhxk123.oss-cn-shanghai.aliyuncs.com/plots/{task_id}_plot1.png'
    task.save()


@shared_task
def ant_colony_tsp(cities, coordinates, task_id, aco_params):
    num_cities = len(cities)
    distance_matrix = create_distance_matrix(coordinates)
    G = nx.Graph()
    # 绘制初始的无向图
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            G.add_edge(cities[i], cities[j], weight=distance_matrix[i][j])
    solver = acopy.Solver(rho=float(aco_params[2]), q=float(aco_params[1]))
    colony = acopy.Colony(alpha=1, beta=3)
    # tour = solver.solve(G, colony, limit=int(aco_params[0]))
    limit = int(aco_params[0])
    limit1 = limit
    task = Schedule.objects.get(id=task_id)
    process1 = 0
    with tqdm(total=limit, desc="Ant Colony Optimization Progress") as pbar:
        for _ in range(limit):
            tour = solver.solve(G, colony, limit=1)  # 每次只进行一次迭代
            process1 += 1/limit1
            task.process1 = process1
            task.save()
            pbar.update(1)  # 更新进度条
    task.result = tour.cost
    task.save()
    plot_path(cities, coordinates, tour.path, task_id)


@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task5(self, task_id, aco_params, ga_params, saa_params):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        file_name = task.image_oss_url
        image_file = bucket.get_object(file_name)
        cities = []
        coordinates = []
        image_file_content = image_file.read().decode('utf-8')
        csv_file = StringIO(image_file_content)
        reader = csv.reader(csv_file, delimiter=';')
        for row in reader:
            cities.append(row[0].strip())  # 去除城市名称两端的空格
            coordinates.append((float(row[1]), float(row[2])))  # 解析城市的经纬度坐标
        chord(
            group(
                ant_colony_tsp.s(cities, coordinates, task_id, aco_params),
                ga_process.s(cities, coordinates, task_id, ga_params),
                ssa_tsp.s(cities, coordinates, task_id, saa_params)
            ),
            update_task_status.s(task_id)
        ).apply_async(queue='low_priority')
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


@shared_task
def update_task_status(task1, task_id):  # 这里的task1是前面函数的返回值组成的resultgroup,所以需要一个参数去接受
    task = Schedule.objects.get(id=task_id)
    task.status = 'completed'
    task.save()


@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task6(self, task_id, aco_params, ga_params, saa_params):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        file_name = task.image_oss_url
        image_file = bucket.get_object(file_name)
        cities = []
        coordinates = []
        image_file_content = image_file.read().decode('utf-8')
        csv_file = StringIO(image_file_content)
        reader = csv.reader(csv_file, delimiter=';')
        for row in reader:
            cities.append(row[0].strip())  # 去空格
            coordinates.append((float(row[1]), float(row[2])))  # 经纬度
        chord(
            group(
                ant_colony_tsp.s(cities, coordinates, task_id, aco_params),
                ga_process.s(cities, coordinates, task_id, ga_params),
                ssa_tsp.s(cities, coordinates, task_id, saa_params)
            ),
            update_task_status.s(task_id)
        ).apply_async(queue='high_priority')
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


def evaluate(individual, distance_matrix):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distance_matrix[individual[i]][individual[i + 1]]
    total_distance += distance_matrix[individual[-1]][individual[0]]
    return (total_distance,)


def plot_path2(cities, coordinates, path, task_id):
    plt.figure(figsize=(8, 6))
    plt.scatter(np.array(coordinates)[:, 0], np.array(coordinates)[:, 1], c='blue', marker='o')
    for i, c in enumerate(cities):
        plt.text(coordinates[i][0], coordinates[i][1], c, fontsize=9)
    for i in range(len(path) - 1):
        start_city = path[i]
        end_city = path[i + 1]
        start_index = cities.index(start_city)
        end_index = cities.index(end_city)
        start_coord = coordinates[start_index]
        end_coord = coordinates[end_index]
        plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], c='red', linewidth=1)
    plt.title('GA Best Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    temp_file_path = f'{task_id}_plot.png'
    plt.savefig(temp_file_path)
    plt.close()
    with open(temp_file_path, 'rb') as file:
        bucket.put_object(f'plots/{task_id}_plot2.png', file)
    os.remove(temp_file_path)
    task = Schedule.objects.get(id=task_id)
    task.p2_oss_url = f'https://hxkhxk123.oss-cn-shanghai.aliyuncs.com/plots/{task_id}_plot2.png'
    task.save()


@shared_task
def ga_process(cities, coordinates, task_id, ga_params):
    # 设置适应度
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # 创建染色体
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # 生成初始染色体序列，这个染色体是一个随机的城市序列
    toolbox.register("indices", random.sample, range(len(cities)), len(cities))
    # 初始染色体
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    # 种群
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # 有序交叉
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, distance_matrix=create_distance_matrix(coordinates))
    distance_matrix = create_distance_matrix(coordinates)
    pop = toolbox.population(n=int(ga_params[1]))
    task = Schedule.objects.get(id=task_id)
    limit = int(ga_params[0])
    process2 = 0
    # 运行遗传算法
    for gen in tqdm(range(int(ga_params[0]))):
        process2 += 1/limit
        task.process2 = process2
        task.save()
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=float(ga_params[2]), ngen=1, verbose=False)
    # algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=float(ga_params[2]), ngen=int(ga_params[0]), verbose=True)
    # 选择最优个体
    best_individual = tools.selBest(pop, k=1)[0]
    result = evaluate(best_individual, distance_matrix)

    task.result2 = result[0]
    task.save()
    city_index_to_name = {idx: city for idx, city in enumerate(cities)}
    path_with_names = [city_index_to_name[idx] for idx in best_individual]
    path_with_names.append(path_with_names[0])
    plot_path2(cities, coordinates, path_with_names, task_id)


def objective_function(path, coordinates):
    dist = 0
    path = [int(idx) for idx in path]
    for i in range(len(path) - 1):
        dist += distance(coordinates[path[i]], coordinates[path[i+1]])
    dist += distance(coordinates[path[-1]], coordinates[path[0]])  # 回到起点
    return dist


def plot_path3(cities, coordinates, path, task_id):
    plt.figure(figsize=(8, 6))
    plt.scatter(np.array(coordinates)[:, 0], np.array(coordinates)[:, 1], c='blue', marker='o')
    for i, c in enumerate(cities):
        plt.text(coordinates[i][0], coordinates[i][1], c, fontsize=9)
    for i in range(len(path) - 1):
        start_city = path[i]
        end_city = path[i + 1]
        start_index = cities.index(start_city)
        end_index = cities.index(end_city)
        start_coord = coordinates[start_index]
        end_coord = coordinates[end_index]
        plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], c='red', linewidth=1)
    plt.title('SAA Best Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    temp_file_path = f'{task_id}_plot.png'
    plt.savefig(temp_file_path)
    plt.close()
    with open(temp_file_path, 'rb') as file:
        bucket.put_object(f'plots/{task_id}_plot3.png', file)
    os.remove(temp_file_path)
    task = Schedule.objects.get(id=task_id)
    task.p3_oss_url = f'https://hxkhxk123.oss-cn-shanghai.aliyuncs.com/plots/{task_id}_plot3.png'
    task.save()


@shared_task
def ssa_tsp(cities, coordinates, task_id, saa_params, initial_state=None, final_temp=1):
    if initial_state is None:
        initial_state = list(range(len(coordinates)))
        random.shuffle(initial_state)
    max_iter = int(saa_params[0])
    initial_temp = int(saa_params[1])
    alpha = float(saa_params[2])

    class TSPProblem(Annealer):
        def __init__(self, state):
            self.coordinates = coordinates
            super(TSPProblem, self).__init__(state)

        def move(self):       # 重载move
            a, b = sorted(random.sample(range(len(self.state)), 2))
            self.state[a:b + 1] = reversed(self.state[a:b + 1])

        def energy(self):     # 重载energy
            return sum(distance(self.coordinates[self.state[i]], self.coordinates[self.state[i + 1]]) for i in range(len(self.state) - 1)) + distance(self.coordinates[self.state[-1]], self.coordinates[self.state[0]])
    tsp = TSPProblem(initial_state)
    tsp.set_schedule(tsp.auto(minutes=0.2))
    tqdm_iter = tqdm(total=max_iter, desc="Simulated Annealing Progress")
    task = Schedule.objects.get(id=task_id)
    limit = int(saa_params[0])
    process3 = 0
    for _ in range(max_iter):
        # 更新进度条
        tqdm_iter.update(1)
        tqdm_iter.set_postfix({"Best Energy": tsp.energy()})
        # 进行一次完整的退火操作
        tsp.Tmax = initial_temp
        tsp.Tmin = final_temp
        tsp.steps = 1
        tsp.updates = 1
        tsp.copy_strategy = "slice"
        tsp.anneal()
        process3 += 1/limit
        task.process3 = process3
        task.save()
    tqdm_iter.close()
    tsp.Tmax = initial_temp
    tsp.Tmin = final_temp
    tsp.steps = 1
    tsp.updates = 1
    tsp.copy_strategy = "slice"
    best_state, best_energy = tsp.anneal()
    # best_state, best_energy = tsp.anneal()
    best_path = [cities[i] for i in best_state]
    # best_path = [cities[i] for i in tsp.state]
    task.result3 = best_energy
    # task.result3 = tsp.energy()
    task.save()
    plot_path3(cities, coordinates, best_path, task_id)


@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task7(self, task_id):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        file_name = task.image_oss_url
        image_file = bucket.get_object(file_name)
        image_content = image_file.read()
        image_file.close()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        chain(
            process_image_features.s(image_base64).set(queue='low_priority'),
            classify_image_with_model.s(task_id).set(queue='low_priority'),
            update_task_status.s(task_id).set(queue='low_priority')
        ).apply_async()
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


@shared_task(bind=True, max_retries=1, soft_time_limit=120)
def process_user_task8(self, task_id):
    try:
        task = Schedule.objects.get(id=task_id)
        # 进程信息
        task.process_num = os.getpid()
        task.process_info = multiprocessing.current_process().name
        task.save()
        # 运行
        task.status = 'running'
        task.save()
        file_name = task.image_oss_url
        image_file = bucket.get_object(file_name)
        image_content = image_file.read()
        image_file.close()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        chain(
            process_image_features.s(image_base64).set(queue='high_priority'),
            classify_image_with_model.s(task_id).set(queue='high_priority'),
            update_task_status.s(task_id).set(queue='high_priority')
        ).apply_async()
    except SoftTimeLimitExceeded:
        logger.error(f"Task {self.request.id} for task_id {task_id} reached soft time limit.")
        raise
    except Exception as exc:
        logger.error(f"Task {self.request.id} for task_id {task_id} failed: {exc}")
        self.retry(exc=exc)


@shared_task
def process_image_features(processed_image):
    resnet_model = models.resnet50(pretrained=False)
    resnet_model.load_state_dict(torch.load(f'C:/Users/lenovo/Desktop/private/4/task/schedule/resnet50.pth'))
    resnet_model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 对图像进行预处理
    image_content = base64.b64decode(processed_image)
    image_file = BytesIO(image_content)
    image = Image.open(image_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        features = resnet_model(input_batch)
    probabilities = torch.nn.functional.softmax(features[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()
    return predicted_class_index


@shared_task
def classify_image_with_model(predicted_class_index, task_id):
    print(type(predicted_class_index))
    print(predicted_class_index)
    # predicted_class_index = predicted_class_index[0] # 给chord方法使用
    # 加载标签字典
    with open('C:/Users/lenovo/Desktop/private/4/task/schedule/imagenet_classes.json', 'r') as f:
        imagenet_classes_dict = json.load(f)
    imagenet_classes = list(imagenet_classes_dict.values())
    print(imagenet_classes)
    result = imagenet_classes[predicted_class_index]
    task = Schedule.objects.get(id=task_id)
    task.result = result
    task.save()


@shared_task
def monitor_queues():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    high_threshold = 5  # 阈值为5
    high_queue_length = redis_client.llen('high_priority')
    logger.info(f'high_queue_length is {high_queue_length}')
    if high_queue_length is not None and high_queue_length < high_threshold:
        logger.info("High priority queue is underloaded. Balancing queues.")
        tasks_to_move = high_threshold - high_queue_length
        # 从低优先级队列移动任务到高优先级队列
        for _ in range(tasks_to_move):
            task_data = redis_client.rpoplpush('low_priority', 'high_priority')
            logger.info(f"Task {task_data} moved from low to high priority queue.")
    else:
        logger.info("High priority queue is sufficiently loaded. No action needed.")

