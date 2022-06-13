#include <pract_1.h>
#include <lcd.c>
#use standard_io(a)
#use standard_io(b)
#use standard_io(c)

//variables de entrada
int a0,a1,a2,a3,a4,a5,b3;
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


void main()
{
   lcd_init();//Inicializamos la lcd
   set_tris_a(0xFF);//dipswitch
   set_tris_b(0b00001000); //rb3=entrada
   set_tris_c(0x00);//salida a display
   output_b(0x00);
   output_c(0x00);
   output_a(0x00);
   
   a0=0;a1=0;a2=0;a3=0;a4=0;a5=0;b3=0;
      
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
   
   // Matriz de pesos sinápticos
   // Pares
   float W_1[7]={-2.36275721,0.0655979,-1.99657071,-1.56694457,5.66948898,2.24811854,0.37284792};
   // Impares
   float W_2[7]={-2.36275721,0.0655979,-1.99657071,-1.56694457,5.66948898,2.24811854,0.37284792};
   // Mayores a 5
   float W_3[7]={-2.36275721,0.0655979,-1.99657071,-1.56694457,5.66948898,2.24811854,0.37284792};
   
   //Vector de polarización
   float b_1=-0.59955928;  // Pares
   float b_2=-0.59955928;  // Impares
   float b_3=-0.59955928;  // Mayores a 5
   
   while(TRUE)
   {
      int i,j;
      float sum=0;
      int perceptron=0;
      //leemos la entrada
      
      int num[7]={0,0,0,0,0,0,0};
      
      num[0]=input_state(pin_A0);
      num[1]=input_state(pin_A1);
      num[2]=input_state(pin_A2);
      num[3]=input_state(pin_A3);
      num[4]=input_state(pin_A4);
      num[5]=input_state(pin_A5);
      num[6]=input_state(pin_C7);
 
   }

}

