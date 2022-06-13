#include <PRACTICA_01.h>

#define LCD_ENABLE_PIN  PIN_B0                                  
#define LCD_RS_PIN      PIN_B1                               
#define LCD_RW_PIN      PIN_B2                                
#define LCD_DATA4       PIN_B4                               
#define LCD_DATA5       PIN_B5
#define LCD_DATA6       PIN_B6                          
#define LCD_DATA7       PIN_B7

#include <lcd.c>

//sensor de color (piña o manzana)
//celda de carga (peso)
#use standard_io(a)
#use standard_io(b)
#use standard_io(c)
#use standard_io(d)

//variables de entrada
int a0,a1,a2,a3,a4,a5,b3;
int d0,d1,d2;

// Matriz de pesos sinápticos
float W[7]={0,0,0,0,0,0,0};
float W_pares[7]={-1.36295435,1.42190282,-1.98181122,-0.13389304,5.36990477,2.14970312,-0.99937584};
float W_impares[7]={2.27477113,-0.8870185,1.20372817,0.52014407,-5.97250432,-2.75199001,0.36056867};
float W_mayores_5[7]={20.89385631,0.23185898,4.38026094,-17.5526482,4.5199185,3.8940398,10.48108176};

//Vector de polarización
float b;

//función escalón
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
//Números que se pintan
int numeros(int a0,int a1,int a2,int a3,int a4,int a5,int b3){
   int resultado;
   if(a0==1 && a1==1 && a2==1 && a3==1 && a4==1 && a5==1 && b3==0){
      resultado=0;
   }
   else if (a0==0 && a1==1 && a2==1 && a3==0 && a4==0 && a5==0 && b3==0){
      resultado=1;
   }
   
   else if(a0==1 && a1==1 && a2==0 && a3==1 && a4==1 && a5==0 && b3==1){
      resultado=2;
   }
   
   else if(a0==1 && a1==1 && a2==1 && a3==1 && a4==0 && a5==0 && b3==1){
      resultado=3;
   }
   
   else if(a0==0 && a1==1 && a2==1 && a3==0 && a4==0 && a5==1 && b3==1){
      resultado=4;
   }
   
   else if(a0==1 && a1==0 && a2==1 && a3==1 && a4==0 && a5==1 && b3==1){
      resultado=5;
   }
   else if(a0==1 && a1==0 && a2==1 && a3==1 && a4==1 && a5==1 && b3==1){
      resultado=6;
   }
      
   else if(a0==1 && a1==1 && a2==1 && a3==0 && a4==0 && a5==0 && b3==0){
      resultado=7;
   }
   else if(a0==1 && a1==1 && a2==1 && a3==1 && a4==1 && a5==1 && b3==1){
      resultado=8;
   }
   else if(a0==1 && a1==1 && a2==1 && a3==1 && a4==0 && a5==1 && b3==1){
      resultado=9;
   }
   return resultado;
}

void main(){
   lcd_init();//Inicializamos la lcd
   set_tris_a(0xFF);//dipswitch
   set_tris_b(0b00001000); //rb3=entrada
   set_tris_c(0x00);//salida a display
   set_tris_d(0b00000111);
   output_a(0x00);
   output_b(0x00);
   output_c(0x00);
   output_d(0x00);
   // Números del 0 al 9
   //int digitos[10]={0b00111111,0b00000110,0b01011011,0b01001111,0b01100110,0b01101101,0b01111101,0b00000111,0b01111111,0b01101111};
   
   // Para el producto punto
   int numero_0[7]={1,1,1,1,1,1,0};//0
   int numero_1[7]={0,1,1,0,0,0,0};//1
   int numero_2[7]={1,1,0,1,1,0,1};//2
   int numero_3[7]={1,1,1,1,0,0,1};//3
   int numero_4[7]={0,1,1,0,0,1,1};//4
   int numero_5[7]={1,0,1,1,0,1,1};//5
   int numero_6[7]={1,0,1,1,1,1,1};//6
   int numero_7[7]={1,1,1,0,0,0,0};//7
   int numero_8[7]={1,1,1,1,1,1,1};//8
   int numero_9[7]={1,1,1,1,0,1,1};//9

   
   while(TRUE)
   {
      int res=0,i;
      //Leemos la entrada de datos
      a0=input(PIN_A0);
      a1=input(PIN_A1);
      a2=input(PIN_A2);
      a3=input(PIN_A3);
      a4=input(PIN_A4);
      a5=input(PIN_A5);
      b3=input(PIN_B3);
      d0=input(PIN_D0);
      d1=input(PIN_D1);
      d2=input(PIN_D2);
      // Pares
      if(d0==1 && d1==0 && d2==0){
         for (int i=0;i<7;i++){
            W[i] = W_pares[i];
         }
         b=0.32647933;
         res=1;
      }
      //Impares
      else if(d0==0 && d1==1 && d2==0){
         for (int i=0;i<7;i++){
            W[i] = W_impares[i];
         }
         b=1.13023473;
         res=2;
      }
      //Mayores a 5
      else if(d0==0 && d1==0 && d2==1){
         for (int i=0;i<7;i++){
            W[i] = W_mayores_5[i];
         }
         b=-22.14247551;
         res=3;
      }
      else{
         lcd_gotoxy(1,1);
         printf(lcd_putc,"\fNo existe");
         res=0;
      }

      //RA0
      if(a0==1){
      output_high(PIN_C0);//Enciende C0
      }
      else{output_low(PIN_C0);} //Apaga C0
      //RA1
      if(a1==1){
      output_high(PIN_C1);//Enciende C1
      }
      else{output_low(PIN_C1);}//Apaga C1
      //RA2
      if(a2==1){
      output_high(PIN_C2);//Enciende C2
      }
      else{output_low(PIN_C2);}//Apaga C2
      //RA3
      if(a3==1){
      output_high(PIN_C4);//Enciende C4
      }
      else{output_low(PIN_C4);}//Apaga C4
      //RA4
      if(a4==1){
      output_high(PIN_C5);//Enciende C5
      }
      else{output_low(PIN_C5);}//Apaga C5
      //RA5
      if(a5==1){
      output_high(PIN_C6);//Enciende C6
      }
      else{output_low(PIN_C6);}//Apaga C6
      //RB3
      if(b3==1){
      output_high(PIN_C7);//Enciende C7
      }
      else{output_low(PIN_C7);}//Apaga C7
      
      for (i=0;i<10;i++){ // Recorre a todos los números
         int j,k;
         float sum=0;
         int perceptron=0;
         k = numeros(a0,a1,a2,a3,a4,a5,b3);
      
         switch (k){
         case 0:
         for(j=0;j<7;j++){
               sum += numero_0[j]*W[j];//producto punto
            }
         break;
         case 1:
         for(j=0;j<7;j++){
               sum = sum +numero_1[j]*W[j];//producto punto
         }
         break;
         case 2:
         for(j=0;j<7;j++){
               sum = sum +numero_2[j]*W[j];//producto punto
         } 
         break;
         case 3:
         for(j=0;j<7;j++){
               sum = sum +numero_3[j]*W[j];//producto punto
         }
         break;
         case 4:
         for(j=0;j<7;j++){
               sum = sum +numero_4[j]*W[j];//producto punto
         }
         break;
         case 5:
         for(j=0;j<7;j++){
               sum = sum +numero_5[j]*W[j];//producto punto
         }
         break;
         case 6:
         for(j=0;j<7;j++){
               sum = sum +numero_6[j]*W[j];//producto punto
         } 
         break;
         case 7:
         for(j=0;j<7;j++){
               sum = sum +numero_7[j]*W[j];//producto punto
         }
         break;
         case 8:
         for(j=0;j<7;j++){
               sum = sum +numero_8[j]*W[j];//producto punto
         } 
         break;
         case 9:
         for(j=0;j<7;j++){
               sum = sum +numero_9[j]*W[j];//producto punto
         } 
         break;
         default:
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fNo existe ese #\n");
         break;
         }
         
         perceptron = hardlim(sum+b);
           // Pares
         if(perceptron==1 && res==1){ //se activó, es par
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fSi es par"); 
         }
         else if(perceptron==0 && res==1){
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fNo es par");
         }
         //impares
         else if(perceptron==1 && res==2){
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fSi es impar");
         }
         else if(perceptron==0 && res==2){
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fNo es impar\n");
         }
         //Mayores a 5
         else if(perceptron==1 && res==3){
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fSi es >5");
         }
         else if(perceptron==0 && res==3){
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fNo es >5");
         }
         else{
            lcd_gotoxy(1,1);
            printf(lcd_putc,"\fNo existe");
         }
         delay_ms(1000);
         
         a0=0;a1=0;a2=0;a3=0;a4=0;a5=0;b3=0;
         d0=0;d1=0;d2=0;
         res=0;
         
      }
   }
}
