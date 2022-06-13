#include <Reloj.h> //Nombre del proyecto
#define LCD_RS_PIN      PIN_B1 //pin B1 en la lcd
#define LCD_RW_PIN      PIN_B2 //pin B2 en la lcd
#define LCD_ENABLE_PIN  PIN_B3 //pin B3 en la lcd
#define LCD_DATA4       PIN_B4 //pin B4 en la lcd
#define LCD_DATA5       PIN_B5 //pin B5 en la lcd
#define LCD_DATA6       PIN_B6 //pin B6 en la lcd
#define LCD_DATA7       PIN_B7 //pin B7 en la lcd
#include <lcd.c>  //Incluimos la libreria LCD
#use standard_io(b) //uso de puerto b


//Declaración de variable
int contador = 217; //variable contadora a decrementar
int segundo_izquierda = 0;//segundo 1
int segundo_derecha = 0; //segundo 2
int minuto_izquierda = 0;//minuto 1
int minuto_derecha = 0;//minuto 2
int hora_izquierda = 0;//hora 1
int hora_derecha = 0;//hora 2


//Caracteres animados
int gato1[8]={31,0,31,0,31,0,31,0};
int gato2[8]={15,8,9,8,8,8,15,0};
int gato3[8]={30,2,2,18,10,6,30,0};
int gato4[8]={16,25,22,16,16,22,31,0};
int gato5[8]={16,16,16,16,16,16,16,0};




// Rutina de animación y actualización del reloj
void gato(){
    
    lcd_gotoxy(4,1);
    printf(lcd_putc,"%d%d: %d%d: %d%d", hora_izquierda,hora_derecha,minuto_izquierda,minuto_derecha,segundo_izquierda,segundo_derecha);//escribe la hora
    
    for(int i=0; i<=16;i++){ //recorre la segunda fila de izquierda a derecha
      
      lcd_gotoxy(4,1);
      printf(lcd_putc,"%d%d: %d%d: %d%d", hora_izquierda,hora_derecha,minuto_izquierda,minuto_derecha,segundo_izquierda,segundo_derecha);//escribe la hora
      
      lcd_gotoxy(i,2); //columna, fila
      printf(lcd_putc," %c%c%c%c%c",0,1,2,3,4);
      delay_ms(200);
      
      /*
      lcd_gotoxy(4,1);
      printf(lcd_putc,"%d%d: %d%d: %d%d", hora_izquierda,hora_derecha,minuto_izquierda,minuto_derecha,segundo_izquierda,segundo_derecha);//escribe la hora
      
      lcd_gotoxy(i,2);
      printf(lcd_putc," %c%c",1,2);
      delay_ms(200);
      
      
      lcd_gotoxy(4,1);
      printf(lcd_putc,"%d%d: %d%d: %d%d", hora_izquierda,hora_derecha,minuto_izquierda,minuto_derecha,segundo_izquierda,segundo_derecha);//escribe la hora
      
      
      lcd_gotoxy(i,2);
      printf(lcd_putc," %c%c",2,3);
      delay_ms(200);
      
      
      
      lcd_gotoxy(4,1);
      printf(lcd_putc,"%d%d: %d%d: %d%d", hora_izquierda,hora_derecha,minuto_izquierda,minuto_derecha,segundo_izquierda,segundo_derecha);//escribe la hora
      
      lcd_gotoxy(i,2);
      printf(lcd_putc," %c%c",3,4);
      delay_ms(200);
      
      
      lcd_gotoxy(4,1);
      printf(lcd_putc,"%d%d: %d%d: %d%d", hora_izquierda,hora_derecha,minuto_izquierda,minuto_derecha,segundo_izquierda,segundo_derecha);//escribe la hora
      
      
      lcd_gotoxy(i,2);
      printf(lcd_putc," %c",4);
      delay_ms(200);
      
      */
      
      
      
      
      
   }
    
    
}



//Función de interrupción timer 0
#INT_TIMER0
void timer0_isr(){ //interrupción interna
   contador--; // decrementa el contador
   set_timer0(238); //este es el valor de seteo del timer
   
   if(contador==0){ // Si llega a cero, se cumplió 1 s
      segundo_derecha++; //Cuando el contador llega a 0 transcurre un segundo
      contador=217;// Inicializa el contador para el próximo periodo
      
      if(segundo_derecha == 10) { //si el segundo de la derecha llega a 10
         segundo_derecha = 0; // se reinicia el valor del segundo de la derecha
         segundo_izquierda++;//aumenta el segundo de la izquierda
         
         if(segundo_izquierda == 6){ //si el segundo de la izquierda es 6 se ha cumplido un minuto
            segundo_izquierda = 0; //se reinicia el valor del segundo de la izquierda
            minuto_derecha++; //aumneta en 1 el minuto
            
            if(minuto_derecha == 10)  {//si el minuto de la derecha llega a 10
               minuto_derecha = 0; // se reinicia el valor del minuto de la derecha
               minuto_izquierda++;//el valor del minuto de la izquierda aumenta
               
               if(minuto_izquierda == 6) {//si el valor del minuto de la izquierda llega a 6 ha pasado una hora               
                  minuto_izquierda = 0; //el minuto de la izquierda se reinicia
                  hora_derecha++; //aumenta una hora
                  
                  if(hora_derecha == 10) { //si la hora de la derecha llega a 10
                     hora_izquierda++; //aumenta en 1 la hora de la izquierda
                     hora_derecha = 0;//se reinicia el valor de la derecha
                     
                     if(hora_izquierda==12) { //si el valor de hora izquierda llega a 12
                        hora_izquierda = 0;//se reinicia el reloj 
                     }
                  }
               }
            }
         }
      }
   }
   
   
}


void main() {//Función principal

   
   lcd_init();//Esta función es utilizada para inicializar la lcd en su modo de 8 bits
   // Declaramos los caracteres especiales en la memoria
   lcd_set_cgram_char(0,gato1);
   lcd_set_cgram_char(1,gato2);
   lcd_set_cgram_char(2,gato3);
   lcd_set_cgram_char(3,gato4);
   lcd_set_cgram_char(4,gato5);
   //printf(lcd_putc,"\f");
   lcd_gotoxy(1,1);
   
   set_tris_b(0x00);//puerto b como salidas
   setup_timer_0(RTCC_INTERNAL|RTCC_DIV_256);//reloj interno del micro y el preescalador de 256
   enable_interrupts(INT_TIMER0); // habilitación de las interrupciones de manera interna
   enable_interrupts(GLOBAL);//habilitar todas las interrupciones
   
   
   
   while(TRUE) { //ciclo infinito  
     gato(); //función de animación
   
   }

}





