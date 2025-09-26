import salabim as sim
import ast
import sys
import pandas as pd
import matplotlib as mpl   # matplotlib版本3.7.4
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QSplitter, QWidget, QLabel, QScrollArea, QPushButton, QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt      # PySide6版本6.6.1
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime
import numpy as np
# pyside6需要python3.8.1以上版本
# matplotlib版本需要3.7.4以上

"""
-------------------------------------两个普通函数--------------------------------------
"""


def read_xlsx():  # 读取文件，得到初始的输入数据结构
    df_pp = pd.read_excel("./input_xlsx/PortPetriNet.xlsx")
    df_ta = pd.read_excel("./input_xlsx/TransitionAttribute.xlsx")
    df_pa = pd.read_excel("./input_xlsx/PlaceAttribute.xlsx")
    df_s = pd.read_excel("./input_xlsx/Ships.xlsx", index_col='船舶编号')
    df_tf = pd.read_excel("./input_xlsx/TransitionFlag.xlsx")
    df_sf = pd.read_excel("./input_xlsx/StopFlag.xlsx")

    # 删除第一列（列标签）
    df_pp = df_pp.iloc[:, 1:]
    # 初始化二维数组
    pn_matrix = []
    # 遍历DataFrame的每一行，将矩阵中每条连线情况的list中的元素设置为str类型，方便对后续对±X/Y的判断
    for index, row in df_pp.iterrows():
        # 初始化当前行的列表
        row_list = []
        # 遍历每个单元格
        for item in row:
            # 如果单元格不是NaN（即不是空的）
            if pd.notnull(item):
                # 假设单元格内是一个字符串，将其分割为list
                # 注意：这里假设单元格内的内容已经是字符串格式 '[X,1]'，如果不是，需要先转换为字符串 str(item)
                split_items = str(item).strip('[]').split(',')
                # 将分割后的列表添加到当前行列表中
                row_list.append(split_items)
            else:
                # 如果单元格是空的，添加['0', '0']
                row_list.append(['0', '0'])
        # 将当前行列表添加到二维数组中
        pn_matrix.append(row_list)
    # 最终的pn_matrix的数据类型是：[ [str, str],...... ]

    # 第一行包含转换属性值，第二行包含描述
    transition_attribute_values = df_ta.iloc[0]
    transition_descriptions = df_ta.iloc[1]
    # 初始化最终的列表
    transition_attribute = []
    # 使用enumerate同时遍历行中的每个元素和它们的索引
    for index, item in enumerate(transition_attribute_values):
        # 分割字符串形式的列表（如 '[600,1]'），并处理成两个元素
        elements = item.strip('[]').split(',')
        # 第一个元素转换为整数
        first_element = int(elements[0])
        # 第二个元素保留为字符串
        second_element = elements[1]
        # 从第二行获取描述
        description = transition_descriptions[index]
        # 将整数、字符串和描述添加到最终列表中
        transition_attribute.append([first_element, second_element, description])
    # 最终的transition_attribute类型是：[ [int, str, str], ... ]

    df_s.rename(columns={'进口集装箱X': 'X', '出口集装箱Y': 'Y'}, inplace=True)
    df_s.index = df_s.index.astype(str)  # 将key部分转换成字符串
    ships = df_s.to_dict(orient='index')
    # 最终的ships类型： {str:{'X':int, 'Y':int}, ......}

    '''
    在Python 3.7及以上版本中，字典是按照插入顺序进行遍历的。这是因为从Python 3.7开始，字典被官方宣布为有序的，这意味着当你遍历一个字典时，键值对会按照它们被添加到字典中的顺序返回。
    在Python 3.6中，这个特性是作为字典实现的一个细节被引入的，并且在CPython（Python的默认实现）中保证了顺序，但直到3.7版本才被正式纳入语言规范。
    在Python 3.5及以下版本，字典是无序的，如果你需要有序的数据结构，你应该使用collections.OrderedDict。
    '''

    # 第一行包含地点属性值，第二行包含描述
    place_attribute_values = df_pa.iloc[0]
    place_descriptions = df_pa.iloc[1]
    # 初始化最终的列表
    place_attribute = []
    # 使用zip同时遍历两行数据
    for item, description in zip(place_attribute_values, place_descriptions):
        # 检查单元格是否为空
        if not pd.isnull(item):
            # 使用 ast.literal_eval 安全地将字符串解析为字典
            item_dict = ast.literal_eval(item)
            # 将字典和描述添加到最终列表中
            place_attribute.append([item_dict, description])
    # 最终的place_attribute类型：[ [{str:int}, str文字描述] ,...... ]

    # 将 DataFrame 转换为字典，其中第一列是value，第二列是key，注意和下面的stop_flag正好相反
    transition_flag = pd.Series(df_tf.iloc[:, 0].values, index=df_tf.iloc[:, 1]).to_dict()
    # 数据类型：{str:str ,......}
    # 将 DataFrame 转换为字典，其中第一列是value，第二列是key
    stop_flag = pd.Series(df_sf.iloc[:, 1].values, index=df_sf.iloc[:, 0]).to_dict()

    if len(pn_matrix) != len(transition_attribute):
        print('PN矩阵行数与transition_attribute个数不一致，请检查\n')
    elif len(pn_matrix[0]) != len(place_attribute):
        print('PN矩阵列数与place_attribute个数不一致，请检查\n')
    else:
        print(f'PetriNet矩阵({len(pn_matrix)}行{len(pn_matrix[0])}列)：')
        for row in pn_matrix:
            print(row)
        print(f'\n各Transition耗时及其over条件({len(transition_attribute)}个)：\n', transition_attribute)
        print(f'\n各Place属性及初始token数量({len(place_attribute)}个):\n', place_attribute)
        print(f'\n船只申请简要({len(ships)}艘)(船只编号：进口数， 出口数)：\n', ships)
        print('\n需要记录时刻的Transition：\n', transition_flag)
        print(f'\n标记一个船舶服务结束的状态：\n{stop_flag}\n')

        return pn_matrix, transition_attribute, place_attribute, ships, transition_flag, stop_flag


def initial_time():  # 设置模拟开始时间和时间速度因子
    while True:
        time_str = input("输入模拟开始时间，格式例如（2023-12-01 14:30:00）:\n")
        try:
            initial_start_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').timestamp()
            print("输入的日期和时间是：", datetime.fromtimestamp(initial_start_time).strftime('%Y-%m-%d %H:%M:%S'))
            break
        except ValueError:
            print("输入的日期和时间格式不正确。请按照 '2023-12-01 14:30:00' 的格式输入。\n")

    return initial_start_time


def is_night(timestamp):
    """
    判断给定的时间戳是否为晚上时间（18:00 - 07:00）

    :param timestamp: 时间戳
    :return: 布尔值，如果是晚上时间返回True，否则返回False
    """
    # 将时间戳转换为datetime对象
    dt = datetime.fromtimestamp(timestamp)

    # 获取小时
    hour = dt.hour

    # 判断是否为晚上时间（18:00 - 07:00）
    return 18 <= hour or hour < 7


"""
-------------------------------------两个工具类--------------------------------------
"""


class MyLock(sim.Component):
    id: int

    # 这是salabim24.0的官方建议初始化用法，虽然pycharm可能会警告签名不匹配
    def setup(self, id):
        self.id = id


class InitToStore(sim.Component):
    def process(self):
        return


"""
---------------------------------以下是PetriNet部分代码---------------------------------------------
"""


class Token(sim.Component):

    place_id: int    # 所属place的id
    pid: str         # 所属船只
    which_type: int  # 如果是pub则为 0，公共资源不属于任何船只，否则其余数字分别代表四个子系统的类型

    # 这是salabim24.0的官方建议初始化用法，虽然pycharm可能会警告签名不匹配
    def setup(self, place_id, pid, which_type):
        self.place_id = place_id
        self.which_type = which_type
        if which_type == 0:
            self.pid = 'pub'
        else:
            self.pid = pid


class PlaceStore(sim.Store):

    place_id: int         # Place库所的编号
    is_pub: bool          # 是否存储的是公共资源
    tokens_count: dict    # 为了更快速地获得Place中各token的数量情况，而不需要每次都循环遍历之类的
    describe: str         # 描述
    monitor: sim.Monitor  # 类数据监控

    # 这是salabim24.0的官方建议初始化用法，虽然pycharm可能会警告签名不匹配
    def setup(self, place_id, token_item, describe):  # 初始化，places中要么为pub tokens，要么为空
        self.place_id = place_id
        self.monitor = sim.Monitor(name=f'{place_id}')
        self.tokens_count = token_item
        self.describe = describe
        if token_item.get('pub', 0) != 0:
            self.is_pub = True
            '''
            for num in range(token_item['pub']):
                # 系统初始时place中若有资源，肯定都是pub tokens
                token = Token(place_id=self.place_id, pid='pub', which_type=0)
                init_to_store.to_store(self, token)
            '''
        else:
            self.is_pub = False

    # setup中还没完全创建好place，不能直接在setup中给自身这个place传递资源
    def init_tokens(self, token_item):
        if token_item.get('pub', 0) != 0:
            for num in range(token_item['pub']):
                # 系统初始时place中若有资源，肯定都是pub tokens
                token = Token(place_id=self.place_id, pid='pub', which_type=0)
                init_to_store.to_store(self, token)
            '''
            self.monitor.tally(self.length())
            '''
            self.monitor.tally(sum(self.tokens_count.values()))
        else:
            self.monitor.tally(0)


class ShipProcess(sim.Component):

    pid: str               # 船只process编号
    import_quantity: int   # 进口量
    export_quantity: int   # 出口量
    flag_time: dict        # 所需记录的事件时刻
    transitions: list      # 该船只的一系列transition事件变迁
    is_over: bool          # 时候该船只的所有集装箱业务已经完成
    monitor: sim.Monitor   # 类数据监控

    # 这是salabim24.0的官方建议初始化用法，虽然pycharm可能会警告签名不匹配
    def setup(self, pid, import_quantity, export_quantity):
        self.pid = pid
        self.import_quantity = import_quantity
        self.export_quantity = export_quantity
        self.is_over = False
        self.monitor = sim.Monitor(name=f'{pid}')

        # 初始化船只需要被记录的事件时刻
        self.flag_time = {}
        for k, v in Transition_Flag.items():
            self.flag_time[v] = 'null'

        self.transitions = []
        '''
        # 初始化Transition事件
        for ind, t in enumerate(Transition_Attribute):
            t = TransitionProcess(pid=self.pid, tid=ind, time_cost=t[0], over_condition=t[1])
            self.transitions.append(t)
        '''
    # 如果直接在setup中依次创建事件
    # 先创建的TransitionProcess的那段setup代码还不能访问到ship_process_dict[]
    def create_transitions(self):
        for ind, t in enumerate(Transition_Attribute):
            t = TransitionProcess(pid=self.pid, tid=ind, time_cost=t[0], over_condition=t[1], describe=t[2])
            self.transitions.append(t)


class TransitionProcess(sim.Component):

    pid: str              # 所属船只pid编号
    tid: int              # transition自身编号
    time_cost: int        # transition事件变迁耗时
    over_condition: int   # 结束条件（运行几次后停止）
    my_status: str        # 当前状态：waiting/working/over
    from_places: list     # 需要哪些Places中的资源
    to_places: list       # 会生产资源给哪些Places
    fire_time: int        # 已运行次数
    describe: str         # 作用描述
    monitor: sim.Monitor  # 类数据监控

    def setup(self, pid, tid, time_cost, over_condition, describe):
        self.pid = pid
        self.tid = tid
        self.time_cost = time_cost

        self.my_status = 'waiting'
        self.fire_time = 0
        self.describe = describe
        self.monitor = sim.Monitor(name=f'{pid}_T{tid}')

        if over_condition == 'X':
            self.over_condition = Ships[pid]['X']
        elif over_condition == 'Y':
            self.over_condition = Ships[pid]['Y']
        else:
            self.over_condition = int(over_condition)

        self.from_places = []
        self.to_places = []
        cnt = 0
        for arc in PN_Matrix[tid]:
            int_arc = [0, 0]
            if arc[0] == 'X':
                int_arc[0] = ship_process_dict[pid].import_quantity
            elif arc[0] == '-X':
                int_arc[0] = -ship_process_dict[pid].import_quantity
            elif arc[0] == 'Y':
                int_arc[0] = ship_process_dict[pid].export_quantity
            elif arc[0] == '-Y':
                int_arc[0] = -ship_process_dict[pid].export_quantity
            else:
                int_arc[0] = int(arc[0])

            int_arc[1] = int(arc[1])

            if int_arc[0] < 0:
                # 记录该transition需要消耗的Pj的位置/下标，以及传输线线权重,传输线线类型
                self.from_places.append([cnt, int_arc[0], int_arc[1]])
            elif int_arc[0] > 0:
                # 记录该transition生产token的去向Pj的位置/下标，以及传输线线权重,传输线线类型
                self.to_places.append([cnt, int_arc[0], int_arc[1]])
            cnt += 1

    def process(self):
        self.monitor.tally(self.my_status)
        print(f"Transition {self.tid} of process {self.pid} is running.")

        while True:
            if self.my_status != 'over':  # 事件未over就开始-->检查是否可执行
                printed = False
                while True:
                    if self.my_status == 'over':  # 但这个检查可能陷入无限循环，并其他事件期间可能会将该事件置为over
                        return                    # 这样写可能逻辑上会清晰一点

                    fire = True  # 事件是否可以执行的标志

                    '''
                    # 可以确保检查及消耗tokens的操作的原子性
                    # 但检查的过程不涉及salabim的事件process中断，且对store资源的请求先来先得，所以不加也行
                    self.from_store(all_place_lock)
                    for i in self.from_places:
                        self.from_store(place_lock, filter=lambda x: x.id == i[0])
                    self.to_store(all_place_lock, MyLock(id=0))
                    '''

                    # 检查是否所有所需tokens都满足
                    for i in self.from_places:
                        if i[2] != 0 and place_stores[i[0]].tokens_count.get(f'{self.pid}_{i[2]}', 0) < -i[1]:
                            # 注意这里只需要pid+资源类型，不是pid+tid
                            fire = False
                            break
                        elif i[2] == 0 and place_stores[i[0]].tokens_count.get('pub', 0) < -i[1]:
                            fire = False
                            break
                        # 对于某Place中只剩1个pub token的情况，要优先分配给流程上更靠前的事件transition，以避免死锁
                        elif i[2] == 0 and place_stores[i[0]].tokens_count.get('pub', 0) == 1:
                            for t in range(len(PN_Matrix)):
                                if PN_Matrix[t][i[0]][0] not in ('0', 'X', 'Y', '1') and t < self.tid and not all(ship_process.transitions[t].my_status == 'over' for ship_process in ship_process_dict.values()):
                                    fire = False
                                    if not printed:  # 可能会频繁输出，让其只输出一次就好
                                        print(f'{self.pid} T{self.tid} 请求 {i[0]}号Place最后一个pub_token失败，无优先权限，继续等待')
                                        printed = True
                                    break
                            if not fire:
                                break

                    if not fire:
                        '''
                        for i in self.from_places:
                            self.to_store(place_lock, MyLock(id=i[0]))
                        '''
                        # 如果不是所有条件都满足，就等待一定时间再检查
                        self.hold(1)
                    else:
                        break

                if printed:  # 之前请求过最后一个pub_token失败过，现在请求到了
                    print(f'经过等待，{self.pid} T{self.tid} 请求到了 {i[0]}号Place的pub_token')

                # 消耗tokens资源
                self.my_status = 'working'
                self.monitor.tally(self.my_status)
                for i in self.from_places:
                    if i[2] == 0:
                        for t in range(i[1]):
                            self.from_store(place_stores[i[0]])
                            #place_stores[i[0]].tokens_count['pub'] -= 1
                        place_stores[i[0]].tokens_count['pub'] += i[1]
                        #place_stores[i[0]].monitor.tally(place_stores[i[0]].length())
                        place_stores[i[0]].monitor.tally(sum(place_stores[i[0]].tokens_count.values()))
                        print(f'{self.pid} T{self.tid} 消耗 {i[0]}号Place的 {-i[1]}个 {i[2]}类资源')

                    else:
                        for t in range(i[1]):
                            self.from_store(place_stores[i[0]],
                                            filter=lambda token: token.pid == self.pid and token.which_type == i[2])
                            #place_stores[i[0]].tokens_count[f'{self.pid}_{i[2]}'] -= 1
                        place_stores[i[0]].tokens_count[f'{self.pid}_{i[2]}'] = place_stores[i[0]].tokens_count.get(f'{self.pid}_{i[2]}', 0) + i[1]
                        #place_stores[i[0]].monitor.tally(place_stores[i[0]].length())
                        place_stores[i[0]].monitor.tally(sum(place_stores[i[0]].tokens_count.values()))
                        print(f'{self.pid} T{self.tid} 消耗 {i[0]}号Place的 {-i[1]}个 {i[2]}类资源')

                '''
                for i in self.from_places:
                    self.to_store(place_lock, MyLock(id=i[0]))
                '''

                # 模拟耗时
                '''
                print(f'{self.pid} T{self.tid} 需要耗时 {self.time_cost}')
                if is_night(env.now()):
                    self.hold(int(self.time_cost * 1.2))  # 假设晚上效率比白天低一点
                else:
                    self.hold(self.time_cost)
                '''
                # 测试用
                if int(self.time_cost / 40) >= 1:
                    self.hold(int(self.time_cost / 40))
                else:
                    self.hold(1)

                # 生产tokens资源
                for i in self.to_places:
                    if i[2] == 0:
                        for t in range(i[1]):
                            self.to_store(place_stores[i[0]], Token(place_id=i[0], pid='pub', which_type=0))
                            # place_stores[i[0]].tokens_count['pub'] += 1
                        place_stores[i[0]].tokens_count['pub'] += i[1]
                        # place_stores[i[0]].monitor.tally(place_stores[i[0]].length())
                        place_stores[i[0]].monitor.tally(sum(place_stores[i[0]].tokens_count.values()))
                        print(f'{self.pid} T{self.tid} 生产给 {i[0]}号Place的 {i[1]}个 {i[2]}类资源')
                    else:
                        for t in range(i[1]):
                            self.to_store(place_stores[i[0]], Token(place_id=i[0], pid=self.pid, which_type=i[2]))
                            # place_stores[i[0]].tokens_count[f'{self.pid}_{i[2]}'] = place_stores[i[0]].tokens_count.get(f'{self.pid}_{i[2]}', 0) + 1
                        place_stores[i[0]].tokens_count[f'{self.pid}_{i[2]}'] = place_stores[i[0]].tokens_count.get(f'{self.pid}_{i[2]}', 0) + i[1]
                        # place_stores[i[0]].monitor.tally(place_stores[i[0]].length())
                        place_stores[i[0]].monitor.tally(sum(place_stores[i[0]].tokens_count.values()))
                        print(f'{self.pid} T{self.tid} 生产给 {i[0]}号Place的 {i[1]}个 {i[2]}类资源')

                self.fire_time += 1  # 事件执行次数+1

                # 执行结束，默认重回waiting状态，特殊T变成over
                if self.over_condition == self.fire_time:
                    self.my_status = 'over'
                    self.monitor.tally(self.my_status)
                    print(f'{self.pid} T{self.tid} over')

                    # 对需要纪录时刻的transition做时刻记录
                    if f'T{self.tid}' in Transition_Flag:
                        ship_process_dict[self.pid].flag_time[Transition_Flag[f'T{self.tid}']] = env.now()

                else:
                    self.my_status = 'waiting'
                    self.monitor.tally(self.my_status)

                # 事件执行一次后检查是否所属船舶process该结束
                if f'T{self.tid}' in Stop_Flag:
                    time_to_stop = True
                    for k, v in Stop_Flag.items():
                        if ship_process_dict[self.pid].transitions[int(k[1:])].my_status != 'over':
                            time_to_stop = False
                            break
                    if time_to_stop:
                        for t in ship_process_dict[self.pid].transitions:
                            t.my_status = 'over'
                            t.monitor.tally(t.my_status)
                        ship_process_dict[self.pid].is_over = True
                        print(f'process {self.pid} all transitions is over')
                        print(ship_process_dict[self.pid].flag_time)
                        return

            else:
                return


"""
---------------------------------以下是GUI代码---------------------------------------------
"""


class ResizableCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        self.figure = fig

    def resizeEvent(self, event):
        # 获取宽度和高度，这里height不会改变
        width = self.width()
        # matplotlib图形尺寸是以英寸为单位，dpi是每英寸的点数
        dpi = self.figure.dpi
        self.figure.set_size_inches(width/dpi, self.figure.get_figheight(), forward=True)
        super().resizeEvent(event)
        self.draw()


# 更新CollapsiblePane类，以确保其内容能在展开时自适应大小
class CollapsiblePane(QWidget):
    def __init__(self, title="", parent=None, expanded=False):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.clicked.connect(self.toggle)
        self.content_widget = QWidget()  # 直接使用QWidget容纳内容
        self.content_area = QVBoxLayout(self.content_widget)
        self.content_widget.setLayout(self.content_area)
        self.is_expanded = expanded  # 根据传入的参数设置初始展开状态

        layout = QVBoxLayout(self)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)

        # 根据初始状态设置内容的可见性
        self.content_widget.setVisible(self.is_expanded)
        # 如果初始状态为展开，则需要调整toggle_button的文本或图标
        if self.is_expanded:
            # 调整按钮的文本或图标以反映展开状态
            pass

    def toggle(self):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        # 再次调整按钮的文本或图标以反映当前状态
        if self.is_expanded:
            pass
        else:
            pass


# 主窗口
class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setWindowTitle('Port Simulation Based On PetriNet')
        self.setGeometry(100, 100, 800, 600)

        self.tabWidget = QTabWidget()
        self.setCentralWidget(self.tabWidget)

        # 首先添加框架
        self.shipProcessTab = QWidget()
        self.placeStoreTab = QWidget()
        self.transitionProcessTab = QTabWidget()  # 注意，这里直接使用QTabWidget以便后续添加子标签页

        self.tabWidget.addTab(self.shipProcessTab, "仿真情况")
        self.tabWidget.addTab(self.placeStoreTab, "Places 库所")
        self.tabWidget.addTab(self.transitionProcessTab, "Transitions 事件变迁")

        # 然后填充内容
        self.fillShipProcessTab()
        self.fillPlaceStoreTab()
        self.fillTransitionProcessTab()  # 修正：现在调用填充方法而不是在创建时填充

    def fillShipProcessTab(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 创建右侧的折叠框容器
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 创建图表折叠框
        chart_pane = CollapsiblePane("事件结束时刻对比图", expanded=True)
        # 在这里计算图表的高度，每个横条固定高度为20，上下空间为10
        bar_height = 20
        spacing = 10
        total_bars_height = len(ship_process_dict) * bar_height
        total_height = total_bars_height + (len(ship_process_dict) + 1) * spacing

        # 现在使用计算出的高度来创建图表
        fig = Figure(figsize=(10, total_height), tight_layout=True)
        canvas = ResizableCanvas(fig)
        ax = fig.add_subplot(111)

        # 首先，确定所有类型的事件
        all_events = []
        for ship_process in ship_process_dict.values():
            for event in ship_process.flag_time.keys():
                if event not in all_events:
                    all_events.append(event)

        # 使用色彩映射表来生成足够的颜色
        colormap = plt.colormaps['tab20']
        event_colors = {event: colormap(i / len(all_events)) for i, event in enumerate(all_events)}

        # 设置条形图的位置和宽度
        positions = np.arange(len(ship_process_dict))
        width = 0.1  # 条形的宽度

        # 迭代 ship_process_dict 填充左侧内容并生成条形图数据
        y_ticks = []
        for index, (pid, ship_process) in enumerate(ship_process_dict.items()):
            flag_time = ship_process.flag_time
            y_ticks.append(pid)

            # 创建包含所有时间的字符串，每个时间之间添加换行符
            flag_time_str = f"船舶 {pid}:\n" + "\n".join(
                [f"{event}: {datetime.fromtimestamp(time+start_time)}" for event, time in flag_time.items()])
            flag_time_label = QLabel(flag_time_str)
            flag_time_label.setWordWrap(True)
            left_layout.addWidget(flag_time_label)

            # 为当前的 ship_process 绘制条形图
            event_times = [flag_time.get(event) for event in all_events]
            for i, event_time in enumerate(event_times):
                if event_time:
                    bar = ax.barh(positions[index] + i * width, event_time, height=width,
                                  color=event_colors[all_events[i]], edgecolor='black',
                                  label=all_events[i] if index == 0 else "")
                    # 在每个条形的末端添加文本标签
                    ax.text(event_time + 5, positions[index] + i * width, f"{round(event_time/60, 2)}分钟",
                            va='center', ha='left', fontsize=7)
                else:
                    print('time error!!!')

        # 设置 y 轴标签，标题和图例
        ax.set_yticks(positions + width * len(all_events) / 2)
        ax.set_yticklabels(y_ticks)
        ax.set_xlabel('Time')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=len(all_events), prop={'size': 7})

        # 创建额外信息折叠框
        info_pane = CollapsiblePane("详细信息", expanded=True)
        # 在这里添加额外的信息，例如：
        end_timestamp = max(max(value.flag_time.values()) for value in ship_process_dict.values())
        end_time = datetime.fromtimestamp(end_timestamp+start_time)
        total_import_quantity = sum(value.import_quantity for value in ship_process_dict.values())
        total_export_quantity = sum(value.export_quantity for value in ship_process_dict.values())

        # info_pane.content_area.addWidget(QLabel(f"从{datetime.fromtimestamp(start_time)}开始，到{end_time}，涉及 {len(Ships)} 条船舶的所有集装箱作业完成 \n总进口量:{total_import_quantity}，总出口量:{total_export_quantity}"))
        label = QLabel(
            f"从{datetime.fromtimestamp(start_time)}开始，到{end_time}，涉及 {len(Ships)} 条船舶的所有集装箱作业完成 \n总进口量:{total_import_quantity}，总出口量:{total_export_quantity}")
        label.setAlignment(Qt.AlignCenter)
        # 添加到 info_pane.content_area
        info_pane.content_area.addWidget(label)

        # info_pane.content_area.addWidget(QLabel(f"期间港口总吞吐量：{round((total_import_quantity+total_export_quantity) / end_timestamp *60, 2) } 个集装箱 / 分钟 \n\n"))
        label = QLabel(f"期间港口总吞吐量：{round((total_import_quantity+total_export_quantity) / end_timestamp *60, 2) } 个集装箱 / 分钟 \n\n")
        label.setAlignment(Qt.AlignCenter)
        info_pane.content_area.addWidget(label)

        # 创建 QLabel 并设置文字
        label = QLabel("------------从模拟开始（有船只开始申请进港）时间算起------------")
        # 将文本对齐设置为居中
        label.setAlignment(Qt.AlignCenter)
        # 添加到 info_pane.content_area
        info_pane.content_area.addWidget(label)

        # 依次计算ship_process_dict中每一个values的每一个flag_time中的每一个key event对应value的平均值
        event_sums = {}
        event_counts = {}

        for ship, custom_class_instance in ship_process_dict.items():
            for event, value in custom_class_instance.flag_time.items():
                if event not in event_sums:
                    event_sums[event] = 0
                    event_counts[event] = 0
                event_sums[event] += value
                event_counts[event] += 1

        event_averages = {event: event_sums[event] / event_counts[event] for event in event_sums}

        # 创建QLabel并添加到info_pane.content_area
        for event, average in event_averages.items():
            label = QLabel(f"船舶平均{event}：{round(average/60, 2)} 分钟")
            label.setAlignment(Qt.AlignCenter)
            info_pane.content_area.addWidget(label)

        right_layout.addWidget(info_pane)

        chart_pane.setFixedHeight(total_height * 5)
        chart_pane.content_area.addWidget(canvas)
        right_layout.addWidget(chart_pane)

        # 设置右侧的滚动区域
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_widget = QWidget()  # 创建滚动区域的内容 Widget
        right_scroll_widget.setLayout(right_layout)
        right_scroll_area.setWidget(right_scroll_widget)

        # 创建分隔器并添加左侧和右侧的滚动区域
        splitter = QSplitter(Qt.Horizontal)
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setWidget(left_widget)

        splitter.addWidget(left_scroll_area)
        splitter.addWidget(right_scroll_area)

        # 设置主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)

        # 设置布局到 shipProcessTab 上
        self.shipProcessTab.setLayout(main_layout)

    def fillPlaceStoreTab(self):
        layout = QVBoxLayout()
    
        # 创建一个滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()  # 创建滚动区域的内容 Widget
        scroll_layout = QVBoxLayout(scroll_widget)

        for place_store in place_stores:
            place_pane = CollapsiblePane(
                f"Place ID: {place_store.place_id} —————— 是否存储公共资源: {'是' if place_store.is_pub else '否'} —————— 描述: {place_store.describe}",
                expanded=True)

            # 添加字段显示
            place_pane.content_area.addWidget(QLabel(f"描述: {place_store.describe}"))
            place_pane.content_area.addWidget(QLabel(f"是否存储公共资源: {'是' if place_store.is_pub else '否'}"))

            chart_container = QWidget()
            chart_container_layout = QVBoxLayout(chart_container)
            chart_container.setFixedHeight(300)  # 设置固定高度

            # 绘制图表
            fig, ax = plt.subplots()
            times, values = place_store.monitor.tx()
            ax.plot(times, values)
            # 设置 matplotlib 的中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如，使用 SimHei 字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '−' 显示为方块的问题
            ax.set_title(f"该（{place_store.describe}）Place库所中 总Token数随时间的变化")
            canvas = ResizableCanvas(fig)
            chart_container_layout.addWidget(canvas)
            place_pane.content_area.addWidget(chart_container)

            scroll_layout.addWidget(place_pane)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        self.placeStoreTab.setLayout(layout)  # 设置到placeStoreTab上

    def fillTransitionProcessTab(self):
        for pid, ship_process in ship_process_dict.items():
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # 显示 ShipProcess 的信息
            layout.addWidget(QLabel(f"船舶编号: {ship_process.pid}"))
            layout.addWidget(QLabel(f"进口集装箱数: {ship_process.import_quantity}"))
            layout.addWidget(QLabel(f"出口集装箱数: {ship_process.export_quantity}"))

            scroll_area = QScrollArea(tab)
            scroll_area.setWidgetResizable(True)
            transition_widget = QWidget()
            transitions_layout = QVBoxLayout(transition_widget)

            # 为每个 TransitionProcess 创建一个 CollapsiblePane
            for transition in ship_process.transitions:
                transition_pane = CollapsiblePane(
                    f"事件Transition ID: {transition.tid} —————— 描述：{transition.describe}",
                    expanded=False)
                transitions_layout.addWidget(transition_pane)

                # 显示 TransitionProcess 的属性
                transition_pane.content_area.addWidget(QLabel(f"描述: {transition.describe}"))
                transition_pane.content_area.addWidget(QLabel(f"耗时: {transition.time_cost}"))
                transition_pane.content_area.addWidget(QLabel(f"停止(次数)条件: {transition.over_condition}"))
                transition_pane.content_area.addWidget(
                    QLabel(f"需要哪些Place的token [Place编号，权重，所属系统]:   {transition.from_places}"))
                transition_pane.content_area.addWidget(
                    QLabel(f"产生token给哪些Place [Place编号，权重，所属系统]:   {transition.to_places}"))
                transition_pane.content_area.addWidget(QLabel(f"执行次数: {transition.fire_time}"))

                # 创建图表的容器并设置固定高度
                chart_container = QWidget()
                chart_container_layout = QVBoxLayout(chart_container)
                chart_container.setFixedHeight(300)  # 设置固定高度

                # 绘制图表
                fig, ax = plt.subplots()
                times, values = transition.monitor.tx()

                # 检查是否达到 'over' 状态，并在达到 'over' 后停止绘制
                if 'over' in values:
                    over_index = values.index('over') + 1  # 加1是因为我们想包括over状态的点
                    times = times[:over_index]  # 截断times到over状态出现的位置
                    values = values[:over_index]  # 截断values到over状态出现的位置

                ax.step(times, values, where='post')

                # 创建画布并添加到布局中
                canvas = ResizableCanvas(fig)
                chart_container_layout.addWidget(canvas)
                transition_pane.content_area.addWidget(chart_container)

            scroll_area.setWidget(transition_widget)
            layout.addWidget(scroll_area)

            self.transitionProcessTab.addTab(tab, f"船舶: {pid}")


if __name__ == '__main__':
    # 读取文件，得到一些初始矩阵等
    PN_Matrix, Transition_Attribute, Place_List, Ships, Transition_Flag, Stop_Flag = read_xlsx()
    # 可以设置模拟开始时间
    start_time = initial_time()

    # 创建模拟仿真环境
    env = sim.Environment(trace=False)

    # salabim中对store等容器传递、消耗资源，都必须通过Component
    # 所以这里创建一个Component用来给一些store初始化注入一些资源
    # 并且不可以是data component，必须是有process方法的
    # 所以定义了一个有process的Component子类
    init_to_store = InitToStore()

    # 初始化Places及其中的tokens资源
    place_stores = []
    for ind, item in enumerate(Place_List):
        p = PlaceStore(name=f'place_{ind}', place_id=ind, token_item=item[0], describe=item[1])
        p.init_tokens(item[0])
        place_stores.append(p)

    # 两个Store模拟锁，可以用来保证transition process中检查、消耗tokens操作的原子性
    all_place_lock = sim.Store("all_place_lock")
    place_lock = sim.Store("place_lock")
    init_to_store.to_store(all_place_lock, MyLock(id=0))
    for i in range(len(Place_List)):
        init_to_store.to_store(place_lock, MyLock(id=i))

    # 创建船舶process，以及每条船的事件process
    ship_process_dict = {}
    for k, v in Ships.items():
        s = ShipProcess(pid=k, import_quantity=v['X'], export_quantity=v['Y'])
        ship_process_dict[k] = s
        s.create_transitions()

    env.run()
    print('\n模拟结束，正在加载GUI......')

    # 仿真数据可视化
    #  程序运行过程中打开20个(默认)以上的matplotlib图形窗口，matplotlib会警告
    #  因为这些图形窗口在不被显式关闭的情况下会一直保留在内存中，可能会导致内存消耗过多
    mpl.rcParams['figure.max_open_warning'] = 0  # 设置为0以禁用警告
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
