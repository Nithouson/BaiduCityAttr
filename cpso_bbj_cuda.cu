#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
#define NCITY 357
double beta = 0.4;
char distfile[]="topodist-c357.txt";
char indexfile[]="data-c357-2019-Flat.txt";

///2011-2016 322city
/*
string city_list[]={"济南", "贵阳", "黔南", "六盘水", "南昌", "九江", "鹰潭", "抚州", "上饶",
            "赣州", "重庆", "包头", "鄂尔多斯", "巴彦淖尔", "乌海", "锡林郭勒", "呼和浩特",
            "赤峰", "通辽", "呼伦贝尔", "武汉", "大连", "黄石", "荆州", "襄阳", "黄冈",
            "荆门", "宜昌", "十堰", "随州", "恩施", "鄂州", "咸宁", "孝感", "仙桃", "长沙",
            "岳阳", "衡阳", "株洲", "湘潭", "益阳", "郴州", "福州", "莆田", "三明", "龙岩",
            "厦门", "泉州", "漳州", "上海", "遵义", "黔东南", "湘西", "娄底", "怀化", "常德",
            "天门", "潜江", "滨州", "青岛", "烟台", "临沂", "潍坊", "淄博", "东营", "聊城",
            "菏泽", "枣庄", "德州", "宁德", "威海", "柳州", "南宁", "桂林", "贺州", "贵港",
            "深圳", "广州", "宜宾", "成都", "绵阳", "广元", "遂宁", "巴中", "内江", "泸州",
            "南充", "德阳", "乐山", "广安", "资阳", "自贡", "攀枝花", "达州", "雅安", "吉安",
            "昆明", "玉林", "河池", "玉溪", "楚雄", "南京", "苏州", "无锡", "北海", "钦州",
            "防城港", "百色", "梧州", "东莞", "丽水", "金华", "萍乡", "景德镇", "杭州", "西宁",
            "银川", "石家庄", "衡水", "张家口", "承德", "秦皇岛", "廊坊", "沧州", "温州", "沈阳",
            "盘锦", "哈尔滨", "大庆", "长春", "四平", "连云港", "淮安", "扬州", "泰州", "盐城",
            "徐州", "常州", "南通", "天津", "西安", "兰州", "郑州", "镇江", "宿迁", "铜陵",
            "黄山", "池州", "宣城", "淮南", "宿州", "六安", "滁州", "淮北", "阜阳", "马鞍山",
            "安庆", "蚌埠", "芜湖", "合肥", "辽源", "松原", "云浮", "佛山", "湛江", "江门",
            "惠州", "珠海", "韶关", "阳江", "茂名", "潮州", "揭阳", "中山", "清远", "肇庆",
            "河源", "梅州", "汕头", "汕尾", "鞍山", "朝阳", "锦州", "铁岭", "丹东", "本溪",
            "营口", "抚顺", "阜新", "辽阳", "葫芦岛", "张家界", "大同", "长治", "忻州", "晋中",
            "太原", "临汾", "运城", "晋城", "朔州", "阳泉", "吕梁", "海口", "万宁", "三亚",
            "新余", "南平", "宜春", "保定", "唐山", "南阳", "新乡", "开封", "焦作", "平顶山",
            "许昌", "永州", "吉林", "铜川", "安康", "宝鸡", "商洛", "渭南", "汉中", "咸阳",
            "榆林", "石河子", "庆阳", "定西", "武威", "酒泉", "张掖", "嘉峪关", "台州", "衢州",
            "宁波", "眉山", "邯郸", "邢台", "伊春", "大兴安岭", "黑河", "鹤岗", "七台河", "绍兴",
            "嘉兴", "湖州", "舟山", "平凉", "天水", "白银", "吐鲁番", "昌吉", "哈密", "阿克苏",
            "克拉玛依", "博尔塔拉", "齐齐哈尔", "佳木斯", "牡丹江", "鸡西", "绥化", "乌兰察布",
            "兴安", "大理", "昭通", "红河", "曲靖", "丽江", "金昌", "陇南", "临沧", "济宁",
            "泰安", "莱芜", "双鸭山", "日照", "安阳", "驻马店", "信阳", "鹤壁", "周口", "商丘",
            "洛阳", "漯河", "濮阳", "三门峡", "阿勒泰", "喀什", "和田", "亳州", "吴忠", "固原",
            "延安", "邵阳", "通化", "白山", "白城", "甘孜", "铜仁", "安顺", "毕节", "文山", "保山",
            "阿坝", "拉萨", "乌鲁木齐", "石嘴山", "凉山", "中卫", "巴音郭楞", "来宾", "北京",
            "延边", "塔城", "黔西南", "海东", "克孜勒苏", "那曲", "林芝", "玉树"};
*/
///2017-2019 357city
string city_list[]={"济南", "贵阳", "黔南", "六盘水", "南昌", "九江", "鹰潭", "抚州", "上饶",
            "赣州", "重庆", "包头", "鄂尔多斯", "巴彦淖尔", "乌海", "阿拉善", "锡林郭勒",
            "呼和浩特", "赤峰", "通辽", "呼伦贝尔", "武汉", "大连", "黄石", "荆州", "襄阳",
            "黄冈", "荆门", "宜昌", "十堰", "随州", "恩施", "鄂州", "咸宁", "孝感", "仙桃",
            "长沙", "岳阳", "衡阳", "株洲", "湘潭", "益阳", "郴州", "福州", "莆田", "三明",
            "龙岩", "厦门", "泉州", "漳州", "上海", "遵义", "黔东南", "湘西", "娄底", "怀化",
            "常德", "天门", "潜江", "滨州", "青岛", "烟台", "临沂", "潍坊", "淄博", "东营",
            "聊城", "菏泽", "枣庄", "德州", "宁德", "威海", "柳州", "南宁", "桂林", "贺州",
            "贵港", "深圳", "广州", "宜宾", "成都", "绵阳", "广元", "遂宁", "巴中", "内江",
            "泸州", "南充", "德阳", "乐山", "广安", "资阳", "自贡", "攀枝花", "达州", "雅安",
            "吉安", "昆明", "玉林", "河池", "玉溪", "楚雄", "南京", "苏州", "无锡", "北海",
            "钦州", "防城港", "百色", "梧州", "东莞", "丽水", "金华", "萍乡", "景德镇", "杭州",
            "西宁", "银川", "石家庄", "衡水", "张家口", "承德", "秦皇岛", "廊坊", "沧州", "温州",
            "沈阳", "盘锦", "哈尔滨", "大庆", "长春", "四平", "连云港", "淮安", "扬州", "泰州",
            "盐城", "徐州", "常州", "南通", "天津", "西安", "兰州", "郑州", "镇江", "宿迁",
            "铜陵", "黄山", "池州", "宣城", "淮南", "宿州", "六安", "滁州", "淮北", "阜阳",
            "马鞍山", "安庆", "蚌埠", "芜湖", "合肥", "辽源", "松原", "云浮", "佛山", "湛江",
            "江门", "惠州", "珠海", "韶关", "阳江", "茂名", "潮州", "揭阳", "中山", "清远",
            "肇庆", "河源", "梅州", "汕头", "汕尾", "鞍山", "朝阳", "锦州", "铁岭", "丹东",
            "本溪", "营口", "抚顺", "阜新", "辽阳", "葫芦岛", "张家界", "大同", "长治", "忻州",
            "晋中", "太原", "临汾", "运城", "晋城", "朔州", "阳泉", "吕梁", "海口", "万宁",
            "琼海", "三亚", "儋州", "新余", "南平", "宜春", "保定", "唐山", "南阳", "新乡",
            "开封", "焦作", "平顶山", "许昌", "永州", "吉林", "铜川", "安康", "宝鸡", "商洛",
            "渭南", "汉中", "咸阳", "榆林", "石河子", "庆阳", "定西", "武威", "酒泉", "张掖",
            "嘉峪关", "台州", "衢州", "宁波", "眉山", "邯郸", "邢台", "伊春", "大兴安岭", "黑河",
            "鹤岗", "七台河", "绍兴", "嘉兴", "湖州", "舟山", "平凉", "天水", "白银", "吐鲁番",
            "昌吉", "哈密", "阿克苏", "克拉玛依", "博尔塔拉", "齐齐哈尔", "佳木斯", "牡丹江",
            "鸡西", "绥化", "乌兰察布", "兴安", "大理", "昭通", "红河", "曲靖", "丽江", "金昌",
            "陇南", "临夏", "临沧", "济宁", "泰安", "莱芜", "双鸭山", "日照", "安阳", "驻马店",
            "信阳", "鹤壁", "周口", "商丘", "洛阳", "漯河", "濮阳", "三门峡", "阿勒泰", "喀什",
            "和田", "亳州", "吴忠", "固原", "延安", "邵阳", "通化", "白山", "白城", "甘孜", "铜仁",
            "安顺", "毕节", "文山", "保山", "东方", "阿坝", "拉萨", "乌鲁木齐", "石嘴山", "凉山",
            "中卫", "巴音郭楞", "来宾", "北京", "日喀则", "伊犁", "延边", "塔城", "五指山",
            "黔西南", "海西", "海东", "克孜勒苏", "那曲", "林芝", "玉树", "五家渠", "崇左",
            "普洱", "济源", "西双版纳", "德宏", "文昌", "怒江", "迪庆", "甘南", "陵水", "澄迈",
            "海南", "山南", "昌都", "乐东", "临高", "海北", "黄南", "保亭", "神农架", "果洛",
            "白沙", "阿里", "阿拉尔", "图木舒克"};

#endif // NATIONAL

char outfile[]="NI251.txt";
FILE* out;

///Constants
double pi=acos(-1);

///Reverse Gravity Model Parameters
int Dims = 2 * NCITY - 1; //前l项：推力 后l-1项：吸引力 设第一个城市吸引力=推力
float Search_index[NCITY*NCITY];
int Topo_dist[NCITY*NCITY];
float* d_search_index;
int* d_topo_dist;
double* d_beta;

///PSO Parameters
int Npar = 4096;// Number of particles each swarm
int Maxiter = 10000;  //Maximum Iteration
double Xmin = 0.01;
double Xmax = 10000;
double Xrandmin = 50;
double Xrandmax = 500;
double alpha = 0.75;
double pjump = 0.001;
#define SCALE 0.01
#define CDIMS 30
#define RDIMS 23
int K=int(ceil(1.0*Dims/CDIMS));

#define THREADS_PER_BLOCK 32

//Generate Random N(0,1) Number
double Box_Muller()
{
    double s1=1.0*rand()/RAND_MAX;
    double s2=1.0*rand()/RAND_MAX;
    double r=cos(2*pi*s1)*sqrt(-2*log(s2));
    return r;
}

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
            double predv = SCALE*Push[fcity]*Attr[tcity]/pow(dist,beta);
            SSE += (actv-predv)*(actv-predv);
        }
    }
    double RMSE = sqrt(SSE/(NCITY*(NCITY-1)));
    return RMSE;
}

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
            double predv = SCALE*Push[fcity]*Attr[tcity]/powf(dist,(*pbeta));
            SSE += (actv-predv)*(actv-predv);
        }
    }
    cost[index]=sqrt(SSE/(NCITY*(NCITY-1)));
    return;
}

__global__ void PCost_CUDA(double* pars, double* cost,float* search_index,int* topo_dist, double* pbeta, int* sid, double* gbest)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double fullpar[2*NCITY-1];
    for(int d=0;d<2*NCITY-1;d++)
    {
        fullpar[d]=gbest[d];
    }
    for(int d=0;d<CDIMS;d++)
    {
        fullpar[CDIMS*(*sid)+d]=pars[index*CDIMS+d];
    }

    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=fullpar[c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=fullpar[NCITY+c-1];
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
            double predv = SCALE*Push[fcity]*Attr[tcity]/powf(dist,(*pbeta));
            SSE += (actv-predv)*(actv-predv);
        }
    }
    cost[index]=sqrt(SSE/(NCITY*(NCITY-1)));
    return;
}

__global__ void RCost_CUDA(double* pars, double* cost,float* search_index,int* topo_dist, double* pbeta, int* sid, double* gbest)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double fullpar[2*NCITY-1];
    for(int d=0;d<2*NCITY-1;d++)
    {
        fullpar[d]=gbest[d];
    }
    for(int d=0;d<RDIMS;d++)
    {
        fullpar[CDIMS*(*sid)+d]=pars[index*RDIMS+d];
    }

    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=fullpar[c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=fullpar[NCITY+c-1];
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
            double predv = SCALE*Push[fcity]*Attr[tcity]/powf(dist,(*pbeta));
            SSE += (actv-predv)*(actv-predv);
        }
    }
    cost[index]=sqrt(SSE/(NCITY*(NCITY-1)));
    return;
}


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
            double predv = SCALE*Push[fcity]*Attr[tcity]/powf(dist,beta);
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

int VecDims(int sid)
{
    if(sid==K-1) return RDIMS;
    else return CDIMS;
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
    fprintf(out,"alpha:%f pjump:%f K:%d CDIMS:%d RDIMS:%d scale:%f\n",alpha,pjump,K,CDIMS,RDIMS,SCALE);
    fprintf(out,"Threads_per_block:%d index:%s\n",THREADS_PER_BLOCK,indexfile);

    #ifdef BBJ
    fprintf(out,"BBJ alpha:%f pjump:%f\n",alpha,pjump);
    #endif //BBJ

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

    cudaMalloc((void**) &d_search_index, sizeof(float)*NCITY*NCITY);
    cudaMalloc((void**) &d_topo_dist, sizeof(int)*NCITY*NCITY);
    cudaMemcpy(d_search_index,Search_index,sizeof(float)*NCITY*NCITY,cudaMemcpyHostToDevice);
    cudaMemcpy(d_topo_dist,Topo_dist,sizeof(int)*NCITY*NCITY,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_beta, sizeof(double));
    cudaMemcpy(d_beta,&beta,sizeof(double),cudaMemcpyHostToDevice);


    clock_t tdata=clock();
    cout<<"Data Preparation finished in "<<(tdata-tstart)/(double)CLOCKS_PER_SEC<<"s"<<endl;

    ///System Initialization
    srand(time(NULL));

    double** QPar=new double*[Npar];
    double** QPbest=new double*[Npar];
    double** PPar=new double*[Npar*K];
    double** PPbest=new double*[Npar*K];
    double QPbest_cost[Npar];
    double QGbest[Dims];  //全局最优解
    double QGbest_cost;
    double PPbest_cost[Npar*K];
    double PGbest[Dims];
    double PGbest_cost[K];
    double cur_cost;
    int QGbest_id,PGbest_id[K];

    double* LQPar=new double[Npar*Dims];
    double* LPPar=new double[Npar*CDIMS];
    double cost[Npar];
    double* DQPar;
    double* DPPar;
    double* DPGbest;
    int* d_swarmid;
    double* d_cost;
    cudaMalloc((void **) &DQPar, sizeof(double)*Npar*Dims);
    cudaMalloc((void **) &DPPar, sizeof(double)*Npar*CDIMS);
    cudaMalloc((void **) &DPGbest, sizeof(double)*Dims);
    cudaMalloc((void **) &d_swarmid, sizeof(int));
    cudaMalloc((void **) &d_cost, sizeof(double)*Npar);

    for(int i=0;i<Npar;i++)
    {
        QPar[i]=new double[Dims];
        QPbest[i]=new double[Dims];
    }

    for(int i=0;i<Npar;i++)
    {
        for(int d=0;d<Dims;d++)
        {
            QPar[i][d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
            QPbest[i][d]=QPar[i][d];
        }
    }

    for(int d=0;d<Dims;d++)
    {
        QGbest[d]=QPbest[0][d];
    }
    QGbest_cost = Cost(QPbest[0]);
    QGbest_id=0;

    for(int s=0;s<K;s++)
    {
        for(int p=0;p<Npar;p++)
        {
            PPar[s*Npar+p]=new double[VecDims(s)];
            PPbest[s*Npar+p]=new double[VecDims(s)];
            for(int d=0;d<VecDims(s);d++)
            {
                PPar[s*Npar+p][d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
                PPbest[s*Npar+p][d]=PPar[s*Npar+p][d];
            }

        }
    }

    for(int s=0;s<K;s++)
    {
        for(int d=0;d<VecDims(s);d++)
        {
            PGbest[s*CDIMS+d]=PPar[s*Npar][d];
        }
    }

    PGbest_cost[0]=Cost(PGbest);
    PGbest_id[0]=0;
    for(int s=1;s<K;s++)
    {
        PGbest_cost[s]=PGbest_cost[0];
        PGbest_id[s]=0;
    }

    clock_t tsysinit=clock();
    cout<<"System Init finished in:"<<(tsysinit-tdata)/(double)CLOCKS_PER_SEC<<"s"<<endl;

    double rjump,sigma;
    int Crossid;
    int iter = 0;
    while(iter < Maxiter)
    {
        ///K P-swarms
        for(int s=0;s<K;s++)
        {
            for(int p=0;p<Npar;p++)
            {
                for(int d=0;d<VecDims(s);d++)
                {
                    LPPar[p*VecDims(s)+d]=PPar[s*Npar+p][d];
                }
            }
            cudaMemcpy(DPPar,LPPar, Npar*VecDims(s)*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_swarmid,&s,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(DPGbest,PGbest,Dims*sizeof(double),cudaMemcpyHostToDevice);
            if(s==K-1)
            {
                RCost_CUDA<<<(Npar+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>
                (DPPar,d_cost,d_search_index,d_topo_dist,d_beta,d_swarmid,DPGbest);
            }
            else
            {
                PCost_CUDA<<<(Npar+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>
                (DPPar,d_cost,d_search_index,d_topo_dist,d_beta,d_swarmid,DPGbest);
            }
            cudaMemcpy(cost, d_cost, Npar*sizeof(double), cudaMemcpyDeviceToHost);

            for(int p=0;p<Npar;p++)
            {
                cur_cost=cost[p];

                if(iter==0)
                {
                    PPbest_cost[s*Npar+p]=cur_cost;
                }
                else if(cur_cost < PPbest_cost[s*Npar+p])
                {
                    for(int d=0;d<VecDims(s);d++)
                    {
                        PPbest[s*Npar+p][d] = PPar[s*Npar+p][d];
                    }
                    PPbest_cost[s*Npar+p] = cur_cost;
                }

                if(cur_cost < PGbest_cost[s])
                {
                    for(int d=0;d<VecDims(s);d++)
                    {
                        PGbest[s*CDIMS+d]=PPar[s*Npar+p][d];
                    }
                    PGbest_cost[s]= cur_cost;
                    PGbest_id[s]=p;
                }
            }

            fprintf(out,"Iter:%d Swarm:%d PCost:%f\n",iter,s,PGbest_cost[s]);
            cout<<"Iter:"<<iter<<" Swarm:"<<s<<" PCost:"<<PGbest_cost[s]<<endl;

            for(int p=0;p<Npar;p++)
            {
                for(int d=0;d<VecDims(s);d++)
                {
                    sigma=abs(PGbest[s*CDIMS+d]-PPar[s*Npar+p][d]);
                    PPar[s*Npar+p][d]=PGbest[s*CDIMS+d]+alpha*sigma*Box_Muller();

                    rjump=1.0*rand()/RAND_MAX;
                    if(rjump<pjump)
                    {
                        PPar[s*Npar+p][d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
                    }

                    if(PPar[s*Npar+p][d] > Xmax)
                    {
                        PPar[s*Npar+p][d] = Xmax;
                    }
                    else if(PPar[s*Npar+p][d] < Xmin)
                    {
                        PPar[s*Npar+p][d] = Xmin;
                    }
                }
            }
        }

        ///Exchange
        Crossid=rand()%(Npar/2);
        while(Crossid==QGbest_id)
        {
            Crossid=rand()%(Npar/2);
        }
        for(int d=0;d<Dims;d++)
        {
            QPar[Crossid][d]=PGbest[d];
        }

        ///Q-swarm

        for(int p=0;p<Npar;p++)
        {
            for(int d=0;d<Dims;d++)
            {
                LQPar[p*Dims+d]=QPar[p][d];
            }
        }
        cudaMemcpy(DQPar,LQPar, Npar*Dims*sizeof(double), cudaMemcpyHostToDevice);
        Cost_CUDA<<<(Npar+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(DQPar,d_cost,d_search_index,d_topo_dist,d_beta);
        cudaMemcpy(cost, d_cost, Npar*sizeof(double), cudaMemcpyDeviceToHost);

        for(int p=0;p<Npar;p++)
        {
            cur_cost = cost[p];

            if(iter==0)
            {
                QPbest_cost[p]=cur_cost;
            }
            else if (cur_cost < QPbest_cost[p])
            {
                for(int d=0;d<Dims;d++)
                {
                    QPbest[p][d] = QPar[p][d];
                }
                QPbest_cost[p] = cur_cost;
            }

            if(cur_cost < QGbest_cost)
            {
                for(int d=0;d<Dims;d++)
                {
                    QGbest[d] = QPar[p][d];
                }
                QGbest_cost = cur_cost;
                QGbest_id=p;
            }
        }

        fprintf(out,"Iter:%d QCost:%f\n",iter,QGbest_cost);
        cout<<"Iter:"<<iter<<" QCost:"<<QGbest_cost<<endl;

        for(int p=0;p<Npar;p++)
        {
            for(int d=0;d<Dims;d++)
            {

                sigma=abs(QGbest[d]-QPar[p][d]);
                QPar[p][d]=QGbest[d]+alpha*sigma*Box_Muller();
                rjump=1.0*rand()/RAND_MAX;
                if(rjump<pjump)
                {
                    QPar[p][d]=(1.0*rand()/RAND_MAX)*(Xrandmax-Xrandmin)+Xrandmin;
                }

                if(QPar[p][d] > Xmax)
                {
                    QPar[p][d] = Xmax;
                }
                else if(QPar[p][d] < Xmin)
                {
                    QPar[p][d] = Xmin;
                }
            }
        }

        ///Exchange
        for(int s=0;s<K;s++)
        {
            Crossid=rand()%(Npar/2);
            while(Crossid==PGbest_id[s])
            {
                Crossid=rand()%(Npar/2);
            }
            for(int d=0;d<VecDims(s);d++)
            {
                PPar[s*Npar+Crossid][d]=QGbest[s*CDIMS+d];
            }
        }

        iter += 1;
    }

    double Push[NCITY],Attr[NCITY];
    for(int c=0;c<NCITY;c++)
    {
        Push[c]=QGbest[c];
        if(c==0)
        {
            Attr[0]=Push[0];
        }
        else
        {
            Attr[c]=QGbest[NCITY+c-1];
        }
    }
    for(int c=0;c<NCITY;c++)
    {
        fprintf(out,"%s Push:%f Attr:%f\n",city_list[c].c_str(),Push[c],Attr[c]);
    }
    fprintf(out,"RMSE:%f Rsquare:%f\n",Cost(QGbest),Rsquare(QGbest));
    cout<<"Cost:"<<Cost(QGbest)<<" R^2:"<<Rsquare(QGbest)<<endl;

    for(int p=0;p<Npar;p++)
    {
        delete[](QPar[p]);
        delete[](QPbest[p]);
    }
    for(int p=0;p<Npar*K;p++)
    {
        delete[](PPar[p]);
        delete[](PPbest[p]);
    }
    delete[] QPar;
    delete[] QPbest;
    delete[] PPar;
    delete[] PPbest;

    delete[] LQPar;
    delete[] LPPar;
    cudaFree(d_cost);
    cudaFree(d_beta);
    cudaFree(DQPar);
    cudaFree(DPPar);
    cudaFree(DPGbest);
    cudaFree(d_swarmid);
    cudaFree(d_search_index);
    cudaFree(d_topo_dist);

    clock_t tfinish=clock();
    cout<<"Iteration:"<<(tfinish-tsysinit)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"Total time:"<<(tfinish-tstart)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    fprintf(out,"Total time:%f s\n",(tfinish-tstart)/(double)CLOCKS_PER_SEC);
    fclose(out);
    return 0;
}
