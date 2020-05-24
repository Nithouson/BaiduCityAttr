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
double beta = 0.4;
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

char outfile[]="UR008.txt";
FILE* out;

///Reverse Gravity Model Parameters
int Dims = 2 * NCITY - 1; //ǰl����� ��l-1������� ���һ������������=����
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

    out=fopen(outfile,"w");
    time_t now = time(0); //��ǰϵͳʱ��
    char* dt = ctime(&now); //ת��Ϊ�ַ���
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

    double Gbest[Dims];  //ȫ�����Ž�
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
