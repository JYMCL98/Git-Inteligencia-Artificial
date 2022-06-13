//JOSÉ GUADALUPE SÁNCHEZ VELÁZQUEZ 7° 6
//ALGORITMO PERCEPTRON PARA LA INDENTIFICACIÓN DE NÚMEROS PARES, IMPARES Y MAYORES A 5
#include <IA_P1.h>
#include<lcd.c>

int numeros[10][8]={{1,1,1,1,1,1,0,0}, //#0
                    {0,1,1,0,0,0,0,1}, //#1
                    {1,1,0,1,1,0,1,2}, //#2
                    {1,1,1,1,0,0,1,3}, //#3
                    {0,1,1,0,0,1,1,4}, //#4
                    {1,0,1,1,0,1,1,5}, //#5
                    {1,0,1,1,1,1,1,6}, //#6
                    {1,1,1,0,0,0,0,7}, //#7
                    {1,1,1,1,1,1,1,8}, //#8
                    {1,1,1,1,0,1,1,9}}; //#9
int q=0;
//Números par
float W1[7]={-0.82929156,1.18682705,-1.56890603,-0.18473025,6.43727846,2.23622148,-1.11155999};
float b1=-0.5976918;
//Números impares
float W3[7]={20.89385631,0.23185898,4.38026094,-17.5526482,4.5199185,3.8940398,10.48108176};
float b3=-22.14247551;
//Números mayores a 5
float W2[7]={1.44932157,-1.99287137,2.72580164,1.03251848,-6.44314757,-2.74752946,-0.09115493};
float b2=0.13636565;


int hardlim(float n)
{
   int value;
   
   if(n>0){value=1;}
   else{value=0;}
   
   return value;
}
void mensaje_par()
{
lcd_gotoxy(1,1);printf(lcd_putc," NUMEROS PARES  ");
lcd_gotoxy(1,2);printf(lcd_putc,"    CON IA      ");
}
void mensaje_impar()
{
lcd_gotoxy(1,1);printf(lcd_putc,"NUMEROS IMPARES ");
lcd_gotoxy(1,2);printf(lcd_putc,"    CON IA      ");
}
void mensaje_m5()
{
lcd_gotoxy(1,1);printf(lcd_putc,"  NUMEROS > 5   ");
lcd_gotoxy(1,2);printf(lcd_putc,"    CON IA      ");
}
void mensaje_n()
{
lcd_gotoxy(1,1);printf(lcd_putc," NINGUNA OPCION ");
lcd_gotoxy(1,2);printf(lcd_putc,"  SELECCIONADA  ");
}

void entradas(int q)
{
      float W[7],b;
      if(q==1)//Número pares
      {
         for(int k=0;k<7;k++){W[k]=W1[k];}
         b=b1;   
      }
      else if(q==2)//Numeros impares
      {
         for(int k=0;k<7;k++){W[k]=W2[k];}
         b=b2;
      }
      else if(q==3)//Numeros mayores a 5
      {  
         for(int k=0;k<7;k++){W[k]=W3[k];}
         b=b3;
      }

      int entrada[7]={0,0,0,0,0,0,0};
      
      entrada[0]=input_state(pin_A0);
      entrada[1]=input_state(pin_A1);
      entrada[2]=input_state(pin_A2);
      entrada[3]=input_state(pin_A3);
      entrada[4]=input_state(pin_A4);
      entrada[5]=input_state(pin_A5);
      entrada[6]=input_state(pin_C7);
      int n=0,i=0,j=0;
      
      for(i=0;i<10;i++)
      {
         n=0;
         for(j=0;j<7;j++)
         {
            if (numeros[i][j]==entrada[j])
            {
               n=n+1;
            }
            else
            {
               n=0;
            }
         }
         if(n==7){break;}
      }
      if (n==7)
      {
         lcd_gotoxy(1,1);
         printf(lcd_putc,"Numero %d ",numeros[i][7]);
         output_bit(pin_C0,entrada[0]);
         output_bit(pin_C1,entrada[1]);
         output_bit(pin_C2,entrada[2]);
         output_bit(pin_C3,entrada[3]);
         output_bit(pin_C4,entrada[4]);
         output_bit(pin_C5,entrada[5]);
         output_bit(pin_C6,entrada[6]);
         
         float sum=0;
         int perceptron=0;   
         j=0;
         while(j<7)
         {
            sum=sum+entrada[j]*W[j];
            j++;
         }
         perceptron=hardlim(sum+b);
         
         lcd_gotoxy(1,2);
         if(perceptron==1)
         {
            switch (q)
            {
            case 1:
               printf(lcd_putc,"Si es par  ");
               break;
            case 2:
               printf(lcd_putc,"Si es impar");
            break;
            case 3:
               printf(lcd_putc,"Si es > a 5");
            break;
            }

         }
         else
         {
            switch (q)
            {
            case 1:
               printf(lcd_putc,"No es par  ");
               break;
            case 2:
               printf(lcd_putc,"No es impar");
            break;
            case 3:
               printf(lcd_putc,"No es > a 5");
            break;
            }

         }
      
      }
      else
      {
         lcd_gotoxy(1,1);
         printf(lcd_putc,"No existe       ");
         lcd_gotoxy(1,2);
         printf(lcd_putc,"               ");
         output_C(0b00);
      }
}
void main()
{
   set_tris_A(0x3F);
   set_tris_B(0x3F);
   set_tris_C(0x7F);
   set_tris_D(0x00);
   
   lcd_init();
   
   int B_1=0,B_11=1;
   int B_2=0,B_21=1;
   int B_3=0,B_31=1;
   
   while(TRUE)
   {
      B_1=input_state(pin_B0);
      B_2=input_state(pin_B1);
      B_3=input_state(pin_B2);
      
      if(B_1!=B_11 || B_2!=B_21 || B_3!=B_31)
      {
         printf(lcd_putc,"\f");
         if(B_1==1 && B_2==0 && B_3==0)//Número pares
         {
            mensaje_par();
            delay_ms(1000);
            printf(lcd_putc,"\f");
            q=1;     
         }
         else if(B_1==0 && B_2==1 && B_3==0)//Numeros impares
         {
            mensaje_impar();
            delay_ms(1000);
            printf(lcd_putc,"\f");
            q=2;
         }
         else if(B_1==0 && B_2==0 && B_3==1)//Numeros mayores a 5
         {  
            mensaje_m5();
            delay_ms(1000);
            printf(lcd_putc,"\f");
            q=3;
         }
         else if(B_1==0 && B_2==0 && B_3==0)
         {
            mensaje_n();
            q=4;
         }
         else
         {
            lcd_gotoxy(1,1);printf(lcd_putc,"  SELECCION NO  ");
            lcd_gotoxy(1,2);printf(lcd_putc,"     VALIDA     ");
            q=4;
         }
      }
      
      if (q!=4)
      {

      entradas(q);
      }
      
      B_11=B_1;
      B_21=B_2;
      B_31=B_3;
   }
}
