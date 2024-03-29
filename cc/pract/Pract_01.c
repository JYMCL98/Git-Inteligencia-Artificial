#include <Pract_01.h>

//Definimos los pines de la LCD que vamos a ocupar para comodidad
#define LCD_ENABLE_PIN  PIN_B0                                  
#define LCD_RS_PIN      PIN_B1                               
#define LCD_RW_PIN      PIN_B2                                
#define LCD_DATA4       PIN_B4                               
#define LCD_DATA5       PIN_B5
#define LCD_DATA6       PIN_B6                          
#define LCD_DATA7       PIN_B7

#include<lcd.c>

// Establecemos los puertos a ocupar
#use standard_io(a)
#use standard_io(b)
#use standard_io(c)
#use standard_io(d)

// Funci�n Hardlim
int hardlim(float n){
   int value;
   if(n>0){
      value=1;
   }
   else{
      value=0;
   }
   return value;
}

// Matriz de 10(Numero de combinaciones para los numeros del 1-9) por 8 (Combinaciones de cada numero)
int P[10][7]={{1,1,1,1,1,1,0},  // #0
              {0,1,1,0,0,0,0},  // #1
              {1,1,0,1,1,0,1},  // #2
              {1,1,1,1,0,0,1},  // #3
              {0,1,1,0,0,1,1},  // #4
              {1,0,1,1,0,1,1},  // #5
              {1,0,1,1,1,1,1},  // #6
              {1,1,1,0,0,0,0},  // #7
              {1,1,1,1,1,1,1},  // #8
              {1,1,1,1,0,1,1}};  // #9
int aux=0;

// Matrices de pesos sin�pticos
float W1[7]={-2.0995771,0.22821014,-1.1819343,-1.39593523,7.49902654,2.73781325,0.13927451};
float W2[7]={3.83011134,-0.94126381,3.24334515,3.00442704,-7.2976491,-3.65744617,-1.02459802};
float W3[7]={19.82199677,1.58512717,6.36077531,-17.25688612,4.4639811,3.0588916,10.11007424};

// Valores iniciales
float W[7];
float b;
int neuro_1=0,neuro_2=0,neuro_3=0;
int neuro_01=1,neuro_02=1,neuro_03=1;
int num[7]={0,0,0,0,0,0,0};


// Funci�n que relaciona el switch con el display de 7 segmentos
void obt_numero(){

      num[0] = input(PIN_A0);
      num[1] = input(PIN_A1);
      num[2] = input(PIN_A2);
      num[3] = input(PIN_A3);
      num[4] = input(PIN_A4);
      num[5] = input(PIN_A5);
      num[6] = input(PIN_B3);
      
      output_bit(PIN_C0, num[0]);
      output_bit(PIN_C1, num[1]);
      output_bit(PIN_C2, num[2]);
      output_bit(PIN_C3, num[3]);
      output_bit(PIN_C4, num[4]);
      output_bit(PIN_C5, num[5]);
      output_bit(PIN_C6, num[6]);
}

void Pros_neu(int aux){
      // Caso de n�mero pares
      if(aux==1){
         for(int k=0; k<7; k++){
            W[k] = W1[k];
         }
         b=-0.1515541;   
      }
      // Caso de numeros impares
      else if(aux==2){
         for(int k=0;k<7;k++){
            W[k] = W2[k];
         }
         b=0.47042927;
      }
      // Caso de n�meros mayores a 5
      else if(aux==3){  
         for(int k=0;k<7;k++){
            W[k] = W3[k];
         }
         b=-23.05911308;
      }
      
      obt_numero();
      
      int n=0,i=0,j=0;
      
      for(i=0;i<10;i++){
         n=0;
         for(j=0;j<7;j++){
            if (P[i][j]==num[j]){
               n += 1;
            }
            else
            {
               n=0;
            }
         }
         if(n==7){
            break;
         }
      }
      if(n==7){
         float sum=0;
         int perceptron=0;   
         j=0;
         
         while(j<7){
            sum += num[j]*W[j];
            j++;
         }
         
         perceptron = hardlim(sum+b);
         
         if(perceptron==1){
            switch (aux){
            case 1:
               lcd_gotoxy(1,1);
               printf(lcd_putc,"\fSi es par");
               delay_ms(1000);
               break;
            case 2:
               lcd_gotoxy(1,1);
               printf(lcd_putc,"\fSi es impar");
               delay_ms(1000);
            break;
            case 3:
               lcd_gotoxy(1,1);
               printf(lcd_putc,"\fSi es > 5");
               delay_ms(1000);
            break;
            }
            
         }
         else{
            switch (aux){
            case 1:
               lcd_gotoxy(1,1);
               printf(lcd_putc,"\fNo es par");
               delay_ms(1000);
               break;
            case 2:
               lcd_gotoxy(1,1);
               printf(lcd_putc,"\fNo es impar");
               delay_ms(1000);
            break;
            case 3:
               lcd_gotoxy(1,1);
               printf(lcd_putc,"\fNo es > 5");
               delay_ms(1000);
            break;
            }
         }
      }
      
      else{
         lcd_gotoxy(1,1);
         printf(lcd_putc,"\fNo existe       ");
         output_C(0b00);
         delay_ms(1000);
      }
}

void main(){

   set_tris_a(0xFF);
   set_tris_b(0b00001000);
   set_tris_c(0x00);
   set_tris_d(0b00000111);

   lcd_init();

   while(TRUE){
      
      obt_numero();
      
      // Elegir el caso de neurona a ocupar
      neuro_1 = input(PIN_D0);
      neuro_2 = input(PIN_D1);
      neuro_3 = input(PIN_D2);
      
      if(neuro_1!=neuro_01 || neuro_2!=neuro_02 || neuro_3!=neuro_03){
         printf(lcd_putc,"\f");
         // Combinacion para n�meros pares
         if(neuro_1==1 && neuro_2==0 && neuro_3==0){
            aux=1;
         }
         // Combinacion para n�meros impares
         else if(neuro_1==0 && neuro_2==1 && neuro_3==0){
            aux=2;
         }
         // Combinacion para n�meros mayores a 5
         else if(neuro_1==0 && neuro_2==0 && neuro_3==1){  
            aux=3;
         }
         // Selecci�n de varios casos
         else{
            lcd_gotoxy(1,1);printf(lcd_putc,"No selecciono");
            lcd_gotoxy(1,2);printf(lcd_putc,"correctamente");
            aux=4;
         }
      }
      
      if (aux!=4){
         Pros_neu(aux);
      }
      
      neuro_01=neuro_1;
      neuro_02=neuro_2;
      neuro_03=neuro_3; 
   }
}

