#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <pthread.h>
#include <semaphore.h>
using namespace std;

//求解城市范围(三选一)
//#define DIAMOND
#define CAPITAL
//#define NATIONAL

#ifdef DIAMOND
#define NCITY 5
double beta = 0.04;
string city_list[]={"北京","上海","广州","武汉","成都"};
char distfile[]="topodist-c5.txt";
char indexfile[]="data-c5-2018.txt";
#endif

#ifdef CAPITAL
#define NCITY 36
double beta = 0.29;
string city_list[]={"北京","上海","天津","重庆","成都","福州","广州","贵阳","哈尔滨","海口",
             "杭州","合肥","呼和浩特","济南","昆明","拉萨","兰州","南昌","南京","南宁",
             "沈阳","石家庄","太原","乌鲁木齐","武汉","西安","西宁","银川","长春",
             "长沙","郑州","大连","宁波","青岛","厦门","深圳"};
char distfile[]="topodist-c36.txt";
char indexfile[]="data-c36-2017-Flat.txt";
#endif // CAPITAL

#ifdef NATIONAL
#define NCITY 360
double beta = 0.35;
char distfile[]="topodist-c360.txt";
char indexfile[]="data-c360-2017.txt";
string city_list[]={"济南", "贵阳", "黔南", "六盘水", "南昌", "九江", "鹰潭", "抚州",
                "上饶", "赣州", "重庆", "包头", "鄂尔多斯", "巴彦淖尔", "乌海", "阿拉善",
                "锡林郭勒", "呼和浩特", "赤峰", "通辽", "呼伦贝尔", "武汉", "大连", "黄石",
                "荆州", "襄阳", "黄冈", "荆门", "宜昌", "十堰", "随州", "恩施", "鄂州",
                "咸宁", "孝感", "仙桃", "长沙", "岳阳", "衡阳", "株洲", "湘潭", "益阳",
                "郴州", "福州", "莆田", "三明", "龙岩", "厦门", "泉州", "漳州", "上海",
                "遵义", "黔东南", "湘西", "娄底", "怀化", "常德", "天门", "潜江", "滨州",
                "青岛", "烟台", "临沂", "潍坊", "淄博", "东营", "聊城", "菏泽", "枣庄",
                "德州", "宁德", "威海", "柳州", "南宁", "桂林", "贺州", "贵港", "深圳",
                "广州", "宜宾", "成都", "绵阳", "广元", "遂宁", "巴中", "内江", "泸州",
                "南充", "德阳", "乐山", "广安", "资阳", "自贡", "攀枝花", "达州", "雅安",
                "吉安", "昆明", "玉林", "河池", "玉溪", "楚雄", "南京", "苏州", "无锡",
                "北海", "钦州", "防城港", "百色", "梧州", "东莞", "丽水", "金华", "萍乡",
                "景德镇", "杭州", "西宁", "银川", "石家庄", "衡水", "张家口", "承德",
                "秦皇岛", "廊坊", "沧州", "温州", "沈阳", "盘锦", "哈尔滨", "大庆", "长春",
                "四平", "连云港", "淮安", "扬州", "泰州", "盐城", "徐州", "常州", "南通",
                "天津", "西安", "兰州", "郑州", "镇江", "宿迁", "铜陵", "黄山", "池州",
                "宣城", "巢湖", "淮南", "宿州", "六安", "滁州", "淮北", "阜阳", "马鞍山",
                "安庆", "蚌埠", "芜湖", "合肥", "辽源", "松原", "云浮", "佛山", "湛江",
                "江门", "惠州", "珠海", "韶关", "阳江", "茂名", "潮州", "揭阳", "中山",
                "清远", "肇庆", "河源", "梅州", "汕头", "汕尾", "鞍山", "朝阳", "锦州",
                "铁岭", "丹东", "本溪", "营口", "抚顺", "阜新", "辽阳", "葫芦岛", "张家界",
                "大同", "长治", "忻州", "晋中", "太原", "临汾", "运城", "晋城", "朔州",
                "阳泉", "吕梁", "海口", "万宁", "琼海", "三亚", "儋州", "新余", "南平",
                "宜春", "保定", "唐山", "南阳", "新乡", "开封", "焦作", "平顶山", "许昌",
                "永州", "吉林", "铜川", "安康", "宝鸡", "商洛", "渭南", "汉中", "咸阳",
                "榆林", "石河子", "庆阳", "定西", "武威", "酒泉", "张掖", "嘉峪关",
                "台州", "衢州", "宁波", "眉山", "邯郸", "邢台", "伊春", "大兴安岭", "黑河",
                "鹤岗", "七台河", "绍兴", "嘉兴", "湖州", "舟山", "平凉", "天水", "白银",
                "吐鲁番", "昌吉", "哈密", "阿克苏", "克拉玛依", "博尔塔拉", "齐齐哈尔",
                "佳木斯", "牡丹江", "鸡西", "绥化", "乌兰察布", "兴安", "大理", "昭通",
                "红河", "曲靖", "丽江", "金昌", "陇南", "临夏", "临沧", "济宁", "泰安",
                "莱芜", "双鸭山", "日照", "安阳", "驻马店", "信阳", "鹤壁", "周口", "商丘",
                "洛阳", "漯河", "濮阳", "三门峡", "阿勒泰", "喀什", "和田", "亳州", "吴忠",
                "固原", "延安", "邵阳", "通化", "白山", "白城", "甘孜", "铜仁", "安顺",
                "毕节", "文山", "保山", "东方", "阿坝", "拉萨", "乌鲁木齐", "石嘴山",
                "凉山", "中卫", "巴音郭楞", "来宾", "北京", "日喀则", "伊犁", "延边",
                "塔城", "五指山", "黔西南", "海西", "海东", "克孜勒苏", "那曲", "林芝",
                "玉树", "五家渠", "香港", "澳门", "崇左", "普洱", "济源", "西双版纳",
                "德宏", "文昌", "怒江", "迪庆", "甘南", "陵水", "澄迈", "海南", "山南",
                "昌都", "乐东", "临高", "海北", "黄南", "保亭", "神农架", "果洛", "白沙",
                "阿里", "阿拉尔", "图木舒克"};
#endif // NATIONAL

#define INIT_V  //速度初始化
//#define CHECK_END  //根据Cost判断终止
//#define RAND  //递减的速度随机量
#define INERTIA //速度惯性项

char outfile[]="RG006.txt";
FILE* out;

///Reverse Gravity Model Parameters
int Dims = 2 * NCITY - 1; //前l项：推力 后l-1项：吸引力 设第一个城市吸引力=推力
double Search_index[NCITY][NCITY];
int Topo_dist[NCITY][NCITY];

///PSO Parameters
double c1 = 2;
double c2 = 2;  // Acceleration constants
#define NPAR 2000 // Number of particles
int Maxiter = 5000;  //Maximum Iteration
double Xmin = 0.1;
double Xmax = 100;
double Xrandmin = 5;
double Xrandmax = 70;
double Vmax = 10;
double wmax = 0.9;
double wmin = 0.4;
int wdesclim = Maxiter; //inertia constant 0.9 to 0.4 desc
double Vrandmax = 5;
double eps = 0.0001;
int epsperiod = 2000;  //End condition: cost decreases less than eps during last epsperiod iterations
double randrad = 5;
int randlim = 0.9*Maxiter;
double bounce = 1;

///Thread Parameters
#define THREADS 48
sem_t control,rcontrol;
pthread_mutex_t mLock;

class Point
{
    public:
        int dims;
        double* pos;

        Point(int _dims,double val=0,bool random=false)
        {
            dims=_dims;
            pos=new double[dims];
            if(!random)
            {
                for(int i=0;i<dims;i++)
                {
                    pos[i]=val;
                }
            }
            else
            {
                for(int i=0;i<dims;i++)
                {
                    pos[i]=val*(2.0*rand()/RAND_MAX-1);
                }
            }
        }

        Point(const Point& p)
        {
            dims=p.dims;
            pos=new double[dims];
            for(int i=0;i<dims;i++)
            {
                pos[i]=p.pos[i];
            }
        }

        const Point& operator=(const Point& s)
        {
            dims=s.dims;
            if(pos == s.pos) return *this;
            if(pos) {delete []pos; pos=NULL;}
            pos = new double[dims];
            for(int i=0;i<dims;i++)
            {
                pos[i]=s.pos[i];
            }
            return *this;
        }

        ~Point()
        {
            if(pos!=NULL)
            {
                delete[] pos;
                pos=NULL;
            }
        }

        int len()const
        {
            return dims;
        }

        double& operator[](int k)const
        {
            return pos[k];
        }

        Point operator+(const Point& opr)
        {
            Point res = Point(dims);
            for(int i=0;i<dims;i++)
            {
                res.pos[i] = pos[i] + opr.pos[i];
            }
            return res;
        }

        Point& operator+=(const Point& opr)
        {
            *this=*this+opr;
            return *this;
        }

        Point operator-(const Point& opr)
        {
            Point res = Point(dims);
            for(int i=0;i<dims;i++)
            {
                res.pos[i] = pos[i] - opr.pos[i];
            }
            return res;
        }

        Point operator*(const double opr)const
        {
            Point res = Point(dims);
            for(int i=0;i<dims;i++)
            {
                res.pos[i] = pos[i] * opr;
            }
            return res;
        }

        friend Point operator*(double opr,const Point& p)
        {
            return p*opr;
        }

        friend Point Filter(const Point&p, double Min, double Max, double bounce = 0)
        {
            Point res=Point(p.dims);
            for(int i=0;i<p.dims;i++)
            {
                if(p.pos[i] > Max)
                {
                    res.pos[i] = Max - bounce;
                }
                else if(p.pos[i] < Min)
                {
                    res.pos[i] = Min + bounce;
                }
                else
                {
                    res.pos[i] = p.pos[i];
                }
            }
            return res;
        }

};
Point Gbest = Point(Dims);
double Gbest_cost;

//Cost Function
double Cost(const Point& par)
{
    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=par[c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=par[NCITY+c-1];
        }
    }
    double SSE = 0;
    for(int fcity=0;fcity<NCITY;fcity++)
    {
        for(int tcity=0;tcity<NCITY;tcity++)
        {
            if(tcity == fcity) continue;
            int dist = Topo_dist[fcity][tcity];
            double actv = Search_index[fcity][tcity];
            double predv = Push[fcity]*Attr[tcity]/pow(dist,beta);
            SSE += (actv-predv)*(actv-predv);
        }
    }
    double RMSE = sqrt(SSE/(NCITY*(NCITY-1)));
    return RMSE;
}

double Rsquare(const Point& par)
{
    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=par[c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=par[NCITY+c-1];
        }
    }
    vector<double>predlist,actlist;
    for(int fcity=0;fcity<NCITY;fcity++)
    {
        for(int tcity=0;tcity<NCITY;tcity++)
        {
            if(tcity == fcity) continue;
            int dist = Topo_dist[fcity][tcity];
            double actv = Search_index[fcity][tcity];
            double predv = Push[fcity]*Attr[tcity]/pow(dist,beta);
            predlist.push_back(predv);
            actlist.push_back(actv);
        }
    }
    double Sx=0,Sy=0,Sxx=0,Sxy=0;
    int n=NCITY*(NCITY-1);
    for(int i=0;i<n;i++)
    {
        Sx+=actlist[i];
        Sy+=predlist[i];
        Sxx+=actlist[i]*actlist[i];
        Sxy+=actlist[i]*predlist[i];
    }
    double b=(n*Sxy-Sx*Sy)/(n*Sxx-Sx*Sx),a=(Sy-b*Sx)/n; //y=bx+a
    vector<double> estlist;
    for(int i=0;i<n;i++)
    {
        double estv=b*actlist[i]+a;
        estlist.push_back(estv);
    }
    double SSE = 0,SST = 0;
    double meanv = Sy/n;
    for(int i=0;i<n;i++)
    {
        SSE += (estlist[i]-predlist[i])*(estlist[i]-predlist[i]);
        SST += (meanv-predlist[i])*(meanv-predlist[i]);
    }
    return 1 - SSE/SST;
}

//递减的惯性权重
double Inertia(int iter)
{
    if(iter > wdesclim)
        return wmin;
    else return wmax - 1.0*iter/wdesclim * (wmax-wmin);
}

//RAND开启时,递减的随机数生成半径
double RandRadius(int iter)
{
    if(iter > randlim)
        return 0;
    else return randrad * (1 - 1.0*iter/randlim);
}

void* PSOThread(void *i)
{
    int npar=int(ceil(NPAR/THREADS));
    vector<Point> Particles;
    vector<Point> Pbest; //各粒子历史最优解
    vector<Point> Veclist;
    double Pbest_cost[npar];
    Point Tbest=Point(Dims);
    double Tbest_cost;
    srand(time(NULL));

    for(int i=0;i<npar;i++)
    {
        Point par=Point(Dims);
        for(int d=0;d<Dims;d++)
        {
            par[d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
        }
        Particles.push_back(par);
    }
    for(int i=0;i<npar;i++)
    {
        Pbest.push_back(Particles[i]);
    }

    Tbest = Pbest[0];  //全局最优解
    Tbest_cost = Cost(Pbest[0]);

    for(int p=0;p<npar;p++)
    {
        Point v=Point(Dims);
        #ifdef INIT_V
        for(int d=0;d<Dims;d++)
        {
            v[d]=(1.0*rand()/RAND_MAX)*(2*Vrandmax)-Vrandmax;
        }
        #endif // INIT_VEC
        Veclist.push_back(v);
    }
    for(int p=0;p<npar;p++)
    {
        double curr_cost = Cost(Particles[p]);
        Pbest_cost[p]=curr_cost;
        if(curr_cost < Tbest_cost)
        {
            Tbest = Particles[p];
            Tbest_cost = curr_cost;
        }
    }

    double r1,r2,RMSE;
    int iter = 0;
    while(iter < Maxiter)
    {
        sem_wait(&control);
        for(int i=0;i<npar;i++)
        {
            r1 = 1.0*rand()/RAND_MAX;
            r2 = 1.0*rand()/RAND_MAX;
            Point vec = c1 * r1 * (Pbest[i]-Particles[i])+ c2 * r2 *(Tbest-Particles[i]);
            #ifdef INERTIA
            vec += Inertia(iter) * Veclist[i];
            #endif // INERTIA
            #ifdef RAND
            if(iter < randlim)
               vec = Point(Dims, RandRadius(iter), true) + vec;
            #endif
            vec = Filter(vec,-Vmax,Vmax);
            Veclist[i] = vec;
            Particles[i] = (Particles[i] + vec);
            Particles[i] = Filter(Particles[i],Xmin,Xmax,bounce);
        }
        for(int i=0;i<npar;i++)
        {
            RMSE=Cost(Particles[i]);
            if (RMSE < Pbest_cost[i])
            {
                Pbest[i] = Particles[i];
                Pbest_cost[i] = RMSE;
                if(Pbest_cost[i]< Tbest_cost)
                {
                    Tbest = Pbest[i];
                    Tbest_cost = Pbest_cost[i];
                }
            }
        }
        pthread_mutex_lock(&mLock);     //加锁
        if (Gbest_cost>Tbest_cost)
        {
            Gbest=Tbest;
            Gbest_cost=Tbest_cost;
        }
        else
        {
            Tbest=Gbest;
            Tbest_cost=Gbest_cost;
        }
        pthread_mutex_unlock(&mLock);   //解锁
        sem_post(&rcontrol);
        iter++;
    }
    return 0;
}

void* PSOMainThread(void *i)
{
    int npar=int(ceil(NPAR/THREADS));
    vector<Point> Particles;
    vector<Point> Pbest; //各粒子历史最优解
    vector<Point> Veclist;
    double Pbest_cost[npar];
    Point Tbest = Point(Dims);
    double Tbest_cost;
    srand(time(NULL));

    for(int i=0;i<npar;i++)
    {
        Point par=Point(Dims);
        for(int d=0;d<Dims;d++)
        {
            par[d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
        }
        Particles.push_back(par);
    }
    for(int i=0;i<npar;i++)
    {
        Pbest.push_back(Particles[i]);
    }

    Tbest = Pbest[0];  //全局最优解
    Tbest_cost = Cost(Pbest[0]);

    for(int p=0;p<npar;p++)
    {
        Point v=Point(Dims);
        #ifdef INIT_V
        for(int d=0;d<Dims;d++)
        {
            v[d]=(1.0*rand()/RAND_MAX)*(2*Vrandmax)-Vrandmax;
        }
        #endif // INIT_VEC
        Veclist.push_back(v);
    }
    for(int p=0;p<npar;p++)
    {
        double curr_cost = Cost(Particles[p]);
        Pbest_cost[p]=curr_cost;
        if(curr_cost < Tbest_cost)
        {
            Tbest = Particles[p];
            Tbest_cost = curr_cost;
        }
    }

    Gbest = Tbest;
    Gbest_cost = Tbest_cost;

    double r1,r2,RMSE;
    int iter = 0;
    while(iter < Maxiter)
    {
        for(int t=0;t<THREADS-1;t++)
        {
            sem_wait(&rcontrol);
        }
        fprintf(out,"Iter:%d Cost:%f\n",iter,Tbest_cost);
        cout<<"Iter:"<<iter<<" Cost:"<<Tbest_cost<<endl;
        for(int i=0;i<npar;i++)
        {
            r1 = 1.0*rand()/RAND_MAX;
            r2 = 1.0*rand()/RAND_MAX;
            Point vec = c1 * r1 * (Pbest[i]-Particles[i])+ c2 * r2 *(Tbest-Particles[i]);
            #ifdef INERTIA
            vec += Inertia(iter) * Veclist[i];
            #endif // INERTIA
            #ifdef RAND
            if(iter < randlim)
               vec = Point(Dims, RandRadius(iter), true) + vec;
            #endif
            vec = Filter(vec,-Vmax,Vmax);
            Veclist[i] = vec;
            Particles[i] = (Particles[i] + vec);
            Particles[i] = Filter(Particles[i],Xmin,Xmax,bounce);
        }
        for(int i=0;i<npar;i++)
        {
            RMSE=Cost(Particles[i]);
            if (RMSE < Pbest_cost[i])
            {
                Pbest[i] = Particles[i];
                Pbest_cost[i] = RMSE;
                if(Pbest_cost[i]< Tbest_cost)
                {
                    Tbest = Pbest[i];
                    Tbest_cost = Pbest_cost[i];
                }
            }
        }
        pthread_mutex_lock(&mLock);     //加锁
        if (Gbest_cost>Tbest_cost)
        {
            Gbest=Tbest;
            Gbest_cost=Tbest_cost;
        }
        else
        {
            Tbest=Gbest;
            Tbest_cost=Gbest_cost;
        }
        pthread_mutex_unlock(&mLock);   //解锁
        for(int t=1;t<=THREADS-1;t++)
        {
            sem_post(&control);
        }
        iter++;
    }
    return 0;
}


int main()
{
    clock_t tstart=clock();

    out=fopen(outfile,"w");
    time_t now = time(0); //当前系统时间
    char* dt = ctime(&now); //转换为字符串
    fprintf(out,"Time: %s\n",dt);
    fprintf(out,"Beta:%f Cities:%d Npar:%d MaxIter:%d Threads:%d\n",beta,NCITY,NPAR,Maxiter,THREADS);
    fprintf(out,"Xmin:%f Xmax:%f Xrandmin:%f Xrandmax:%f\n",Xmin,Xmax,Xrandmin,Xrandmax);
    fprintf(out,"Vmax:%f c1:%f c2:%f bounce:%f\n",Vmax,c1,c2,bounce);
    #ifdef CHECK_END
    fprintf(out,"CHECKEND  eps:%f epsperiod:%d\n",eps,epsperiod);
    #endif // CHECK_END"
    #ifdef RAND
    fprintf(out,"RAND Randlim:%d randrad:%f\n",randlim,randrad);
    #endif // RAND
    #ifdef INIT_V
    fprintf(out,"INIT_V Vrandmax:%f\n",Vrandmax);
    #endif // INIT_VEC
    #ifdef INERTIA
    fprintf(out,"INERTIA wdesclim:%d wmax:%f wmin%f\n",wdesclim,wmax,wmin);
    #endif // INIT_VEC
    ///Data Initialization
    int i,j,d;
    double val;
    FILE* fdist=fopen(distfile,"r");
    while(fscanf(fdist,"%d %d %d",&i,&j,&d)!=EOF)
    {
        Topo_dist[i][j]=d;
    }
    fclose(fdist);
    FILE* findex=fopen(indexfile,"r");
    while(fscanf(findex,"%d %d %lf",&i,&j,&val)!=EOF)
    {
        Search_index[i][j]=val;
    }
    fclose(findex);

    pthread_t thread[THREADS]; // 定义线程的 id 变量
    pthread_attr_t attr;
    void *status;
    pthread_attr_init(&attr);     // 初始化并设置线程为可连接的（joinable）
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    int ret,tid=0;
    sem_init(&control, 0, THREADS-1);  //信号量初始化
    sem_init(&rcontrol, 0, THREADS-1);  //信号量初始化
    pthread_mutex_init(&mLock, NULL);   //初始化互斥量

    ret = pthread_create(&thread[tid], NULL, PSOMainThread, (void*)&tid);
    if (ret != 0)
    {
        fprintf(out,"pthread_create error: error_code=%d\n",ret);
    }
    for(tid = 1;tid < THREADS; ++tid)
    {
        //参数依次是：创建的线程id，线程参数，调用的函数，传入的函数参数
        ret = pthread_create(&thread[tid], NULL, PSOThread, (void*)&tid);
        if (ret != 0)
        {
            fprintf(out,"pthread_create error: error_code=%d\n",ret);
        }
    }
    pthread_attr_destroy(&attr);
    for( tid=0; tid < THREADS; tid++)
    {
        ret = pthread_join(thread[tid], &status);
        if (ret)
        {
            fprintf(out,"Error:unable to join,code=%d\n",ret);
            exit(-1);
        }
    }
    pthread_mutex_destroy(&mLock);
    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=Gbest[c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=Gbest[NCITY+c-1];
        }
    }
    for(int c=0;c<NCITY;c++)
    {
        fprintf(out,"%s Push:%f Attr:%f\n",city_list[c].c_str(),Push[c],Attr[c]);
    }
    fprintf(out,"RMSE:%f Rsquare:%f\n",Cost(Gbest),Rsquare(Gbest));
    cout<<"Cost:"<<Cost(Gbest)<<" R^2:"<<Rsquare(Gbest)<<endl;
    clock_t tfinish=clock();
    fprintf(out,"Total time:%f s\n",(tfinish-tstart)/(double)CLOCKS_PER_SEC);
    time_t tend = time(0); //当前系统时间
    char* dtend = ctime(&tend); //转换为字符串
    fprintf(out,"Time: %s\n",dtend);
    fclose(out);
    return 0;
}
