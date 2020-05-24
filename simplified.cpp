#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
using namespace std;

//�����з�Χ(��ѡһ)
//#define DIAMOND
//#define CAPITAL
#define NATIONAL

#ifdef DIAMOND
#define NCITY 5
double beta = 0.04;
string city_list[]={"����","�Ϻ�","����","�人","�ɶ�"};
char distfile[]="topodist-c5.txt";
char indexfile[]="data-c5-2018.txt";
#endif

#ifdef CAPITAL
#define NCITY 38
double beta = 0.28;
string city_list[]={"����","�Ϻ�","���","����","�ɶ�","����","����","����","������","����",
             "����","�Ϸ�","���ͺ���","����","����","����","����","�ϲ�","�Ͼ�","����",
             "����","ʯ��ׯ","̫ԭ","��³ľ��","�人","����","����","����","����",
             "��ɳ","֣��","����","����","�ൺ","����","����","���","����"};
char distfile[]="topodist-c38.txt";
char indexfile[]="data-c38-2018.txt";
#endif // CAPITAL

#ifdef NATIONAL
#define NCITY 359
double beta = 0.35;
char distfile[]="topodist-c359.txt";
char indexfile[]="data-c359-2018-Flat.txt";
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
                "����", "����", "����", "����", "����", "����", "����", "��ɽ",
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

//#define INIT_V  //�ٶȳ�ʼ��
//#define CHECK_END  //����Cost�ж���ֹ
#define RAND  //�ݼ����ٶ������
//#define INERTIA //�ٶȹ�����

///Reverse Gravity Model Parameters
int Dims = 2 * NCITY - 1; //ǰl����� ��l-1������� ���һ������������=����
double Search_index[NCITY][NCITY];
int Topo_dist[NCITY][NCITY];

///PSO Parameters
double c1 = 2;
double c2 = 2;  // Acceleration constants
int Npar = 1024;// Number of particles
int Maxiter = 1000;  //Maximum Iteration
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

//Cost Function
//__global__
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
            int dist = Topo_dist[fcity][tcity];
            double actv = Search_index[fcity][tcity];
            double predv = Push[fcity]*Attr[tcity]/pow(dist,beta);
            SSE += (actv-predv)*(actv-predv);
        }
    }
    double RMSE = sqrt(SSE/(NCITY*(NCITY-1)));
    return RMSE;
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

int main()
{
    clock_t tstart=clock();

    time_t now = time(0); //��ǰϵͳʱ��
    char* dt = ctime(&now); //ת��Ϊ�ַ���
    cout<<"Time:"<<dt<<endl;
    cout<<"Beta:"<<beta<<" Cities:"<<NCITY<<" N:"<<Npar<<" MaxIter:"<<Maxiter<<endl;
    cout<<"Xmin:"<<Xmin<<" Xmax:"<<Xmax<<" Xrandmin:"<<Xrandmin<<" Xrandmax:"<<Xrandmax<<endl;
    cout<<"Vmax:"<<Vmax<<" wdesclim:"<<wdesclim<<" wmax:"<<wmax<<" wmin:"<<wmin<<endl;
    cout<<"c1:"<<c1<<" c2:"<<c2<<" bounce:"<<bounce<<endl;
    #ifdef CHECK_END
    cout<<"CHECKEND"<<" eps:"<<eps<<" epsperiod:"<<epsperiod<<endl;
    #endif // CHECK_END"
    #ifdef RAND
    cout<<"RAND:"<<" Randlim:"<<randlim<<" randrad:"<<randrad<<endl;
    #endif // RAND
    #ifdef INIT_V
    cout<<"INIT_V:"<<" Vrandmax:"<<Vrandmax<<endl<<endl;
    #endif // INIT_V

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
    clock_t tdata=clock();
    ///System Initialization
    double** Particles=new double*[Npar];
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
    double Gbest[Dims];  //ȫ�����Ž�
    for(int d=0;d<Dims;d++)
    {
        Gbest[d]=Pbest[0][d];
    }
    double Gbest_cost = Cost(Pbest[0]);

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
    for(int p=0;p<Npar;p++)
    {
        double curr_cost = Cost(Particles[p]);
        Pbest_cost[p]=curr_cost;
        if(curr_cost < Gbest_cost)
        {
            for(int d=0;d<Dims;d++)
            {
                Gbest[d]=Particles[p][d];
            }
            Gbest_cost = curr_cost;
        }
    }
    clock_t tsysinit=clock();

    double cur_cost,r1,r2;
    double cur_velocity[Dims];
    #ifdef CHECK_END
    double Last_Cost = 1e8;
    #endif
    int iter = 0;
    while(iter < Maxiter)
    {
        cout<<"Iter:"<<iter<<setiosflags(ios::fixed)<<setprecision(6)<<" Cost:"<<Gbest_cost<<endl;
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

        for(int p=0;p<Npar;p++)
        {
            cur_cost = Cost(Particles[p]);
            if (cur_cost < Pbest_cost[p])
            {
                for(int d=0;d<Dims;d++)
                {
                    Pbest[p][d]=Particles[p][d];
                }
                Pbest_cost[p] = cur_cost;
                if(cur_cost < Gbest_cost)
                {
                    for(int d=0;d<Dims;d++)
                    {
                        Gbest[d]=Particles[p][d];
                    }
                    Gbest_cost = cur_cost;
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
        cout<<city_list[c]<<" Push:"<<Push[c]<<" Attr:"<<Attr[c]<<endl;
    }
    cout<<"RMSE:"<<Cost(Gbest)<<" Rsquare:"<<Rsquare(Gbest)<<endl;
    for(int i=0;i<Npar;i++)
    {
        delete[](Particles[i]);
        delete[](velocity[i]);
        delete[](Pbest[i]);
    }
    delete[] Particles;
    delete[] velocity;
    delete[] Pbest;

    clock_t tfinish=clock();

    cout<<"Total time:"<<(tfinish-tstart)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"Data Prepare:"<<(tdata-tstart)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"System Init:"<<(tsysinit-tdata)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"Iteration:"<<(tfinish-tsysinit)/(double)CLOCKS_PER_SEC<<"s"<<endl;
    return 0;
}
