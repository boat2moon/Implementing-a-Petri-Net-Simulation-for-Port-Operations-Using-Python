# 船舶服务
__系统的PetriNet图（在PortPertNet文件中）__
![img_20.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_20.png)
__PertriNet三要素：   
Transition事件变迁、Place库所存储资源tokens、连线__  
__也可以看作时一个资源流转图__  

__如果非得把多条船只的并发交互过程画出来，理论上也是可以，但源源不断地要进港的n条属性还不同的船呢？__  
__他们的进口量和出口量都不同，要调度的资源也不同，要让PN图的同一条连线传送的资源随着不同船只而变化__  
__模拟程序基于PetriNet的逻辑，但目的是仿真港口，而不是模拟港口的PetriNet，__  
__所以不能完全按照理想的（抽象的）PetriNet图来写代码，不管是PN图还是代码需要做一些额外的处理__  

__如果使用多进程线程，在模拟时间准确性上会有困难，并且运行程序可能需要等待真实时间。__
__借助离散事件系统仿真Salabim库，使用process来模拟每条船的每个事件，用env环境来负责对时钟的准确模拟。__  

__由于salabim或是其他离散事件系统仿真库基本都是基于协程，只能在主进程中运行；__
__然后一些GUI库如PyQT/PySide、Thinter等也只能在主进程中运行__
__但如果不把两者的运行进程/线程分开，GUI是无法实时响应交互的__

__因此想要实时地可视化系统数据，只能通过salabim自带的动画组件，但功能不如专门的GUI库丰富，需要显示的图表又太多，故等程序运行结束后再通过GUI做模块化显示__


# 六个输入数据文件
## 一、Transition
<mark>(TransitionAttribute.xlsx)</mark>  
![img_22.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_22.png)
__编号__  
__[耗时，结束条件]__  
__描述__  

通过process模拟不同的Transition，用tid区分  
over状态的Transition就不再执行逻辑（if）  
比如一条船只申请进港的Transition就是一次性的

```
# Transition类代码
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
    
    def setup(self, pid, tid, time_cost, over_condition, describe):...
    
    def process(self):...
```

状态在程序运行中给定，有专门的变量存储所有pid的所有tid的状态    
一次性的Transition，主动over条件为1  
运行一次后就会变成over状态，状态存入变量，线程关闭  
比如说船舶申请、船舶进港  
大部分中间transition都不会主动over 是一直循环X/Y次(取决于船只进出口量)
主要是开始和结束的Transition需要这个条件来标记一个船舶服务的开始和结束  
比如说船舶离港也是一次性行为

## 二、连线
连线并不专门写函数或者类表示  
他隐含在PN_Matrix矩阵这个数据结构中  
包括方向权重、类型  
<mark>（PortPertNet.xlsx）</mark>
![img_11.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_11.png)  
![img_12.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_12.png)  
非方阵、稀疏  
Ti--Pj:[a,b]  
第一位正号，代表Ti-->Pj，若为负数代表Pi-->Tj  
第一位数值大小代表权重  
第二位代表传送资源的类别，是为区分船舶服务中的那个子系统在使用该资源  
1号子系统  
2号子系统  
3号子系统  
4号子系统  
0号代表传送的是公共资源，所有pid的所有系统共享的资源  
比如说存储空卡车的Place，所有船pid 的 所有子系统都可以直接调用  
但存储载货卡车的PLace，就必须要区分这是那条船pid的那个子系统在调用了，不能送错地方  

可以看到有两个特殊的X Y, X代表的是船只进口量，Y代表船只出口量
这个在“另外两个输入文件”部分会提到

## 三、Place
<mark>（PlaceAttribute.xlsx）</mark>  
![img_23.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_23.png)
__编号：Pn__  
__Tokens： { pid1_1:n,  pid1_2:n, pid3_1:n }__  
__作用：__  

```
# Place类代码
class PlaceStore(sim.Store):
    place_id: int         # Place库所的编号
    is_pub: bool          # 是否存储的是公共资源
    tokens_count: dict    # 为了更快速地获得Place中各token的数量情况，而不需要每次都循环遍历之类的
    describe: str         # 描述
    monitor: sim.Monitor  # 类数据监控
    
    def setup(self, place_id, token_item, describe):...
    
    def init_tokens(self, token_item):...
```

有一类特殊的存储公共资源的place里只有{ pub：n }  
这由连线的类型决定  
如果连线传输的是0号资源，不会使得pid1_0+1   
而是通过代码判断给pub+1

这样可以做到并发事件的过程中，资源不会混乱，
服务于不同的pid的不同子系统的资源不会“上错菜”  


## 四、另外三个输入文件
<mark>ships.xlsx</mark>    
![img_15.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_15.png)  
船舶id__________________________X__________________________Y  
进口集装箱即代表着这条船要运多少个集装箱过来，卸船系统需要卸载多少个集装箱  
出口集装箱代表着船要从这个港口装走多少货物再离港，也意味着要从内陆集箱到港口的量  
这两个参数比较重要  
会影响许多中间步骤连线的权重  
而开始阶段和结束阶段的部分则更加重要：  
船只申请Transition，产生的是Y个集箱需求    
卸船系统结束的Transition，需要一次性消耗X个“已结束一次卸船”的信号，然后产出Y个装船指令  
船舶离港也需要一次消耗Y个“已结束一次装船”的信号  

![img_11.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_11.png)  
![img_12.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_12.png)  
比如上面提到的（PortPertNet.xlsx）文件中连线权重有XY来标记的  
代表着对应与当前的船舶服务来说（进程），这条连线的权重取决于当前船舶的进口出口集装箱数  
  
  
<mark>StopFlag.xlsx</mark>  
![img_16.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_16.png)  
标记一个船舶服务即一个进程什么时候停止  
假设这里的T28代表船舶离港  
T29代表提箱结束,该船只的进口货物已全部从港口被提走  

船舶可以离港后就显示输出标记的时间，但是该船只的进程这时候还不能关闭  
因为可能还有一部分没被内陆提走的集装箱带有该船只进程的标识符

<mark>TransitionFlag.xlsx</mark>  
![img_24.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_24.png)
设置一个船舶服务要在哪些transition事件状态over后记录一下时刻  
程序可以输出显示  

```
# 船只服务类代码
class ShipProcess(sim.Component):
    pid: str               # 船只process编号
    import_quantity: int   # 进口量
    export_quantity: int   # 出口量
    flag_time: dict        # 所需记录的事件时刻
    transitions: list      # 该船只的一系列transition事件变迁
    is_over: bool          # 时候该船只的所有集装箱业务已经完成
    monitor: sim.Monitor   # 类数据监控
    
    def setup(self, pid, import_quantity, export_quantity):...
    
    def create_transitions(self):...
```

# 该模拟程序的输出
GUI输出界面  
![img_25.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_25.png)
![img_29.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_29.png)
![img_26.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_26.png)
![img_28.png](https://github.com/boat2moon/Implementing-a-Petri-Net-Simulation-for-Port-Operations-Using-Python/blob/main/Picture/img_28.png)
各个船只的一些事件发生的时间，如进港、离港  
模拟结果的一些数据统计
各个船只的各个Transition事件随时间的状态变化 
系统中各Places资源数量情况随时间的变化 
  
至于模拟排队时延，比如说存储载货重卡的Place，其中的tokens数量长时间高于一个阈值 
就可以代表出现了堵塞现象


# Else
目前，资源上没有冲突的事件可以按照并行计算时间，当然公共资源两者都需要，会有一个事件优先级。
资源上有冲突的事件，因为事件B需要事件A的产出，自然会是一个串行。
对于一种资源，优先请求的事件也会优先获得资源。

关于时间的仿真，不再是通过时间戳浮点数的重复计算(产生精度问题的源头之一)，而是有一个最小时间 单位 步长，是个整数。单位可以自己决定是毫秒、秒、分钟、天都可以。

所以时间的模拟可重复。
准确性取决于输入的参数准不准确，模型的流程设计合不合理。

代码运行已经不再是多进程(影响准确性)，但逻辑上，在时间模拟上还是可以考虑并行事件，比如说事件A或事件B设置的耗时都是2个单位，且可以同时执行，那么系统总时间的模拟结果会是2，或者B事件在系统时间1时才启动，那么系统总时间结果会是3。

如果时间结果与真实不符，那就是模型流程(PetriNet)设计得不够细致，不够合理，而不再是代码的问题了。




