#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#define CUDA

#ifdef CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif // CUDA
using namespace std;

//求解城市范围(三选一)
//#define DIAMOND
//#define CAPITAL
#define NATIONAL

#ifdef DIAMOND
#define NCITY 5
double beta = 0.04;
string city_list[]={"北京","上海","广州","武汉","成都"};
char distfile[]="topodist-c5.txt";
char indexfile[]="data-c5-2018.txt";
#endif

#ifdef CAPITAL
#define NCITY 38
double beta = 0.28;
string city_list[]={"北京","上海","天津","重庆","成都","福州","广州","贵阳","哈尔滨","海口",
             "杭州","合肥","呼和浩特","济南","昆明","拉萨","兰州","南昌","南京","南宁",
             "沈阳","石家庄","太原","乌鲁木齐","武汉","西安","西宁","银川","长春",
             "长沙","郑州","大连","宁波","青岛","厦门","深圳","香港","澳门"};
char distfile[]="topodist-c38.txt";
char indexfile[]="data-c38-2018.txt";
#endif // CAPITAL

#ifdef NATIONAL
#define NCITY 359
double beta = 0.4;
char distfile[]="topodist-c359.txt";
char indexfile[]="data-c359-2018-Flat.txt";
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
                "宣城", "淮南", "宿州", "六安", "滁州", "淮北", "阜阳", "马鞍山",
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

//#define INIT_V  //速度初始化
//#define CHECK_END  //根据Cost判断终止
#define RAND  //递减的速度随机量
//#define INERTIA //速度惯性项

char outfile[]="UR008.txt";
FILE* out;

///Reverse Gravity Model Parameters
int Dims = 2 * NCITY - 1; //前l项：推力 后l-1项：吸引力 设第一个城市吸引力=推力
float Search_index[NCITY*NCITY];
int Topo_dist[NCITY*NCITY];
float* d_search_index;
int* d_topo_dist;
double* d_beta;

///PSO Parameters
double c1 = 2;
double c2 = 2;  // Acceleration constants
int Npar = 1024;// Number of particles
int Maxiter = 20000;  //Maximum Iteration
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

#define THREADS_PER_BLOCK 32

//Cost Function
double Cost(const double* par)
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
            int dist = Topo_dist[fcity*NCITY+tcity];
            double actv = Search_index[fcity*NCITY+tcity];
            double predv = Push[fcity]*Attr[tcity]/pow(dist,beta);
            SSE += (actv-predv)*(actv-predv);
        }
    }
    double RMSE = sqrt(SSE/(NCITY*(NCITY-1)));
    return RMSE;
}

#ifdef CUDA
__global__ void Cost_CUDA(double* pars, double* cost,float* search_index,int* topo_dist, double* pbeta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=pars[index*(2*NCITY-1)+c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=pars[index*(2*NCITY-1)+NCITY+c-1];
        }
    }
    double SSE = 0;
    for(int fcity=0;fcity<NCITY;fcity++)
    {
        for(int tcity=0;tcity<NCITY;tcity++)
        {
            if(tcity == fcity) continue;
            int dist = topo_dist[fcity*NCITY+tcity];
            double actv = search_index[fcity*NCITY+tcity];
            double predv = Push[fcity]*Attr[tcity]/powf(dist,(*pbeta));
            SSE += (actv-predv)*(actv-predv);
        }
    }
    cost[index]=sqrt(SSE/(NCITY*(NCITY-1)));
    return;
}
#endif // CUDA

double Rsquare(const double* par)
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
            int dist = Topo_dist[fcity*NCITY+tcity];
            double actv = Search_index[fcity*NCITY+tcity];
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

int main()
{
    clock_t tstart=clock();

    out=fopen(outfile,"w");
    time_t now = time(0); //当前系统时间
    char* dt = ctime(&now); //转换为字符串
    fprintf(out,"Time: %s\n",dt);
    fprintf(out,"Beta:%f Cities:%d Npar:%d MaxIter:%d\n",beta,NCITY,Npar,Maxiter);
    fprintf(out,"Xmin:%f Xmax:%f Xrandmin:%f Xrandmax:%f\n",Xmin,Xmax,Xrandmin,Xrandmax);
    fprintf(out,"Vmax:%f c1:%f c2:%f bounce:%f\n",Vmax,c1,c2,bounce);
    #ifdef CUDA
    fprintf(out,"Threads_per_block:%d\n",THREADS_PER_BLOCK);
    #endif // CUDA
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
    float val;
    FILE* fdist=fopen(distfile,"r");
    while(fscanf(fdist,"%d %d %d",&i,&j,&d)!=EOF)
    {
        Topo_dist[i*NCITY+j]=d;
    }
    fclose(fdist);
    FILE* findex=fopen(indexfile,"r");
    while(fscanf(findex,"%d %d %f",&i,&j,&val)!=EOF)
    {
        Search_index[i*NCITY+j]=val;
    }
    fclose(findex);
    #ifdef CUDA
    cudaMalloc((void**) &d_search_index, sizeof(float)*NCITY*NCITY);
    cudaMalloc((void**) &d_topo_dist, sizeof(int)*NCITY*NCITY);
    cudaMemcpy(d_search_index,Search_index,sizeof(float)*NCITY*NCITY,cudaMemcpyHostToDevice);
    cudaMemcpy(d_topo_dist,Topo_dist,sizeof(int)*NCITY*NCITY,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_beta, sizeof(double));
    cudaMemcpy(d_beta,&beta,sizeof(double),cudaMemcpyHostToDevice);
    #endif // CUDA

    clock_t tdata=clock();
    cout<<"Data Preparation finished in "<<(tdata-tstart)/(double)CLOCKS_PER_SEC<<"s"<<endl;

    ///System Initialization
    srand(time(NULL));

    double** Particles=new double*[Npar];

    #ifdef CUDA
    double* l_particles=new double[Npar*Dims];
    double* d_particles;
    cudaMalloc((void **) &d_particles, sizeof(double)*Npar*Dims);
    #endif

    double** Pbest=new double*[Npar];
    double** velocity=new double*[Npar];
    for(int i=0;i<Npar;i++)
    {
        Particles[i]=new double[Dims];
        Pbest[i]=new double[Dims];
        velocity[i]=new double[Dims];
    }
    for(int i=0;i<Npar;i++)
    {
        for(int d=0;d<Dims;d++)
        {
            Particles[i][d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
            Pbest[i][d]=Particles[i][d];
        }
    }

    double Pbest_cost[Npar];

    #ifdef CUDA
    double cost[Npar];
    double* d_cost;
    cudaMalloc((void **) &d_cost, sizeof(double)*Npar);
    #endif // CUDA

    double Gbest[Dims];  //全局最优解
    for(int d=0;d<Dims;d++)
    {
        Gbest[d]=Pbest[0][d];
    }
    double Gbest_cost = Cost(Pbest[0]),cur_cost;

    for(int p=0;p<Npar;p++)
    {
        #ifdef INIT_V
        for(int d=0;d<Dims;d++)
        {
            velocity[p][d]=(1.0*rand()/RAND_MAX)*(2*Vrandmax)-Vrandmax;
        }
        #endif // INIT_VEC
        #ifndef INIT_V
        for(int d=0;d<Dims;d++)
        {
            velocity[p][d]=0;
        }
        #endif // INIT_V
    }
    #ifdef CUDA
    for(int p=0;p<Npar;p++)
    {
        for(int d=0;d<Dims;d++)
        {
            l_particles[p*Dims+d]=Particles[p][d];
        }
    }
    cudaMemcpy(d_particles,l_particles, Npar*Dims*sizeof(double), cudaMemcpyHostToDevice);
    Cost_CUDA<<<(Npar+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_particles,d_cost,d_search_index,d_topo_dist,d_beta);
    cudaMemcpy(cost, d_cost, Npar*sizeof(double), cudaMemcpyDeviceToHost);
    #endif
    for(int p=0;p<Npar;p++)
    {
        #ifndef CUDA
        cur_cost = Cost(Particles[p]);
        #else // CUDA
        cur_cost=cost[p];
        #endif
        Pbest_cost[p]=cur_cost;
        if(cur_cost < Gbest_cost)
        {
            for(int d=0;d<Dims;d++)
            {
                Gbest[d]=Particles[p][d];
            }
            Gbest_cost = cur_cost;
        }
    }
    clock_t tsysinit=clock();
    cout<<"System Init finished in:"<<(tsysinit-tdata)/(double)CLOCKS_PER_SEC<<"s"<<endl;

    double r1,r2;
    double cur_velocity[Dims];
    #ifdef CHECK_END
    double Last_Cost = 1e8;
    #endif
    int iter = 0;
    while(iter < Maxiter)
    {
        fprintf(out,"Iter:%d Cost:%f\n",iter,Gbest_cost);
        cout<<"Iter:"<<iter<<" Cost:"<<Gbest_cost<<endl;
        #ifdef CHECK_END
        if(iter % epsperiod == 0)
        {
            if(Gbest_cost > Last_Cost - eps) break;
            Last_Cost = Gbest_cost;
        }
        #endif
        for(int p=0;p<Npar;p++)
        {
            r1 = 1.0*rand()/RAND_MAX;
            r2 = 1.0*rand()/RAND_MAX;
            for(int d=0;d<Dims;d++)
            {
                cur_velocity[d]= c1 * r1 * (Pbest[p][d]-Particles[p][d])+ c2 * r2 *(Gbest[d]-Particles[p][d]);
                #ifdef INERTIA
                cur_velocity[d]+=Inertia(iter) * velocity[p][d];
                #endif // INERTIA
                #ifdef RAND
                if(iter < randlim)
                   cur_velocity[d]+= RandRadius(iter)*(2.0*rand()/RAND_MAX-1);
                #endif
                if(cur_velocity[d] > Vmax)
                {
                    cur_velocity[d]  = Vmax;
                }
                else if(cur_velocity[d] < -Vmax)
                {
                    cur_velocity[d] = -Vmax;
                }
                Particles[p][d]+=cur_velocity[d];
                if(Particles[p][d] > Xmax)
                {
                    Particles[p][d] = Xmax - bounce;
                }
                else if(Particles[p][d] < Xmin)
                {
                    Particles[p][d] = Xmin + bounce;
                }
            }

            for(int d=0;d<Dims;d++)
            {
                velocity[p][d]=cur_velocity[d];
            }

        }
        iter += 1;

        #ifdef CUDA
        for(int p=0;p<Npar;p++)
        {
            for(int d=0;d<Dims;d++)
            {
                l_particles[p*Dims+d]=Particles[p][d];
            }
        }
        cudaMemcpy(d_particles,l_particles, Npar*Dims*sizeof(double), cudaMemcpyHostToDevice);
        Cost_CUDA<<<(Npar+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_particles,d_cost,d_search_index,d_topo_dist,d_beta);
        cudaMemcpy(cost, d_cost, Npar*sizeof(double), cudaMemcpyDeviceToHost);
        #endif
        for(int p=0;p<Npar;p++)
        {
            #ifndef CUDA
            cur_cost = Cost(Particles[p]);
            #else // CUDA
            cur_cost=cost[p];
            #endif
            //if(iter<10&&p<50) cout<<p<<" "<<cur_cost<<endl;
            if (cur_cost < Pbest_cost[p])
            {
                for(int d=0;d<Dims;d++)
                {
                    Pbest[p][d] = Particles[p][d];
                }
                Pbest_cost[p] = cur_cost;
                if(cur_cost < Gbest_cost)
                {
                    for(int d=0;d<Dims;d++)
                    {
                        Gbest[d] = Particles[p][d];
                    }
                    Gbest_cost = cur_cost;
                    //cout<<Gbest_cost<<" "<<Cost(Gbest)<<endl;
                }
            }
        }
    }

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

    for(int p=0;p<Npar;p++)
    {
        delete[](Particles[p]);
        delete[](velocity[p]);
        delete[](Pbest[p]);
    }
    delete[] Particles;
    delete[] velocity;
    delete[] Pbest;
    #ifdef CUDA
    delete[] l_particles;
    cudaFree(d_cost);
    cudaFree(d_beta);
    cudaFree(d_particles);
    cudaFree(d_search_index);
    cudaFree(d_topo_dist);
    #endif // CUDA
    clock_t tfinish=clock();
    cout<<"Iteration:"<<(tfinish-tsysinit)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"Total time:"<<(tfinish-tstart)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    fprintf(out,"Total time:%f s\n",(tfinish-tstart)/(double)CLOCKS_PER_SEC);
    fclose(out);
    return 0;
}
