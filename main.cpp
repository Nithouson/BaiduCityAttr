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

//�����з�Χ(��ѡһ)
//#define DIAMOND
#define CAPITAL
//#define NATIONAL

#ifdef DIAMOND
#define NCITY 5
double beta = 0.04;
string city_list[]={"����","�Ϻ�","����","�人","�ɶ�"};
char distfile[]="topodist-c5.txt";
char indexfile[]="data-c5-2018.txt";
#endif

#ifdef CAPITAL
#define NCITY 36
double beta = 0.29;
string city_list[]={"����","�Ϻ�","���","����","�ɶ�","����","����","����","������","����",
             "����","�Ϸ�","���ͺ���","����","����","����","����","�ϲ�","�Ͼ�","����",
             "����","ʯ��ׯ","̫ԭ","��³ľ��","�人","����","����","����","����",
             "��ɳ","֣��","����","����","�ൺ","����","����"};
char distfile[]="topodist-c36.txt";
char indexfile[]="data-c36-2017-Flat.txt";
#endif // CAPITAL

#ifdef NATIONAL
#define NCITY 360
double beta = 0.35;
char distfile[]="topodist-c360.txt";
char indexfile[]="data-c360-2017.txt";
string city_list[]={"����", "����", "ǭ��", "����ˮ", "�ϲ�", "�Ž�", "ӥ̶", "����",
                "����", "����", "����", "��ͷ", "������˹", "�����׶�", "�ں�", "������",
                "���ֹ���", "���ͺ���", "���", "ͨ��", "���ױ���", "�人", "����", "��ʯ",
                "����", "����", "�Ƹ�", "����", "�˲�", "ʮ��", "����", "��ʩ", "����",
                "����", "Т��", "����", "��ɳ", "����", "����", "����", "��̶", "����",
                "����", "����", "����", "����", "����", "����", "Ȫ��", "����", "�Ϻ�",
                "����", "ǭ����", "����", "¦��", "����", "����", "����", "Ǳ��", "����",
                "�ൺ", "��̨", "����", "Ϋ��", "�Ͳ�", "��Ӫ", "�ĳ�", "����", "��ׯ",
                "����", "����", "����", "����", "����", "����", "����", "���", "����",
                "����", "�˱�", "�ɶ�", "����", "��Ԫ", "����", "����", "�ڽ�", "����",
                "�ϳ�", "����", "��ɽ", "�㰲", "����", "�Թ�", "��֦��", "����", "�Ű�",
                "����", "����", "����", "�ӳ�", "��Ϫ", "����", "�Ͼ�", "����", "����",
                "����", "����", "���Ǹ�", "��ɫ", "����", "��ݸ", "��ˮ", "��", "Ƽ��",
                "������", "����", "����", "����", "ʯ��ׯ", "��ˮ", "�żҿ�", "�е�",
                "�ػʵ�", "�ȷ�", "����", "����", "����", "�̽�", "������", "����", "����",
                "��ƽ", "���Ƹ�", "����", "����", "̩��", "�γ�", "����", "����", "��ͨ",
                "���", "����", "����", "֣��", "��", "��Ǩ", "ͭ��", "��ɽ", "����",
                "����", "����", "����", "����", "����", "����", "����", "����", "��ɽ",
                "����", "����", "�ߺ�", "�Ϸ�", "��Դ", "��ԭ", "�Ƹ�", "��ɽ", "տ��",
                "����", "����", "�麣", "�ع�", "����", "ï��", "����", "����", "��ɽ",
                "��Զ", "����", "��Դ", "÷��", "��ͷ", "��β", "��ɽ", "����", "����",
                "����", "����", "��Ϫ", "Ӫ��", "��˳", "����", "����", "��«��", "�żҽ�",
                "��ͬ", "����", "����", "����", "̫ԭ", "�ٷ�", "�˳�", "����", "˷��",
                "��Ȫ", "����", "����", "����", "��", "����", "����", "����", "��ƽ",
                "�˴�", "����", "��ɽ", "����", "����", "����", "����", "ƽ��ɽ", "���",
                "����", "����", "ͭ��", "����", "����", "����", "μ��", "����", "����",
                "����", "ʯ����", "����", "����", "����", "��Ȫ", "��Ҵ", "������",
                "̨��", "����", "����", "üɽ", "����", "��̨", "����", "���˰���", "�ں�",
                "�׸�", "��̨��", "����", "����", "����", "��ɽ", "ƽ��", "��ˮ", "����",
                "��³��", "����", "����", "������", "��������", "��������", "�������",
                "��ľ˹", "ĵ����", "����", "�绯", "�����첼", "�˰�", "����", "��ͨ",
                "���", "����", "����", "���", "¤��", "����", "�ٲ�", "����", "̩��",
                "����", "˫Ѽɽ", "����", "����", "פ���", "����", "�ױ�", "�ܿ�", "����",
                "����", "���", "���", "����Ͽ", "����̩", "��ʲ", "����", "����", "����",
                "��ԭ", "�Ӱ�", "����", "ͨ��", "��ɽ", "�׳�", "����", "ͭ��", "��˳",
                "�Ͻ�", "��ɽ", "��ɽ", "����", "����", "����", "��³ľ��", "ʯ��ɽ",
                "��ɽ", "����", "��������", "����", "����", "�տ���", "����", "�ӱ�",
                "����", "��ָɽ", "ǭ����", "����", "����", "��������", "����", "��֥",
                "����", "�����", "���", "����", "����", "�ն�", "��Դ", "��˫����",
                "�º�", "�Ĳ�", "ŭ��", "����", "����", "��ˮ", "����", "����", "ɽ��",
                "����", "�ֶ�", "�ٸ�", "����", "����", "��ͤ", "��ũ��", "����", "��ɳ",
                "����", "������", "ͼľ���"};
#endif // NATIONAL

#define INIT_V  //�ٶȳ�ʼ��
//#define CHECK_END  //����Cost�ж���ֹ
//#define RAND  //�ݼ����ٶ������
#define INERTIA //�ٶȹ�����

char outfile[]="RG006.txt";
FILE* out;

///Reverse Gravity Model Parameters
int Dims = 2 * NCITY - 1; //ǰl����� ��l-1������� ���һ������������=����
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

//�ݼ��Ĺ���Ȩ��
double Inertia(int iter)
{
    if(iter > wdesclim)
        return wmin;
    else return wmax - 1.0*iter/wdesclim * (wmax-wmin);
}

//RAND����ʱ,�ݼ�����������ɰ뾶
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
    vector<Point> Pbest; //��������ʷ���Ž�
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

    Tbest = Pbest[0];  //ȫ�����Ž�
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
        pthread_mutex_lock(&mLock);     //����
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
        pthread_mutex_unlock(&mLock);   //����
        sem_post(&rcontrol);
        iter++;
    }
    return 0;
}

void* PSOMainThread(void *i)
{
    int npar=int(ceil(NPAR/THREADS));
    vector<Point> Particles;
    vector<Point> Pbest; //��������ʷ���Ž�
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

    Tbest = Pbest[0];  //ȫ�����Ž�
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
        pthread_mutex_lock(&mLock);     //����
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
        pthread_mutex_unlock(&mLock);   //����
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
    time_t now = time(0); //��ǰϵͳʱ��
    char* dt = ctime(&now); //ת��Ϊ�ַ���
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

    pthread_t thread[THREADS]; // �����̵߳� id ����
    pthread_attr_t attr;
    void *status;
    pthread_attr_init(&attr);     // ��ʼ���������߳�Ϊ�����ӵģ�joinable��
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    int ret,tid=0;
    sem_init(&control, 0, THREADS-1);  //�ź�����ʼ��
    sem_init(&rcontrol, 0, THREADS-1);  //�ź�����ʼ��
    pthread_mutex_init(&mLock, NULL);   //��ʼ��������

    ret = pthread_create(&thread[tid], NULL, PSOMainThread, (void*)&tid);
    if (ret != 0)
    {
        fprintf(out,"pthread_create error: error_code=%d\n",ret);
    }
    for(tid = 1;tid < THREADS; ++tid)
    {
        //���������ǣ��������߳�id���̲߳��������õĺ���������ĺ�������
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
    time_t tend = time(0); //��ǰϵͳʱ��
    char* dtend = ctime(&tend); //ת��Ϊ�ַ���
    fprintf(out,"Time: %s\n",dtend);
    fclose(out);
    return 0;
}
