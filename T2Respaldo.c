#include <T2Respaldo.h>
// Pines para la LCD
#define LCD_RS_PIN      PIN_B1
#define LCD_RW_PIN      PIN_B2
#define LCD_ENABLE_PIN  PIN_B3
#define LCD_DATA4       PIN_B4
#define LCD_DATA5       PIN_B5
#define LCD_DATA6       PIN_B6
#define LCD_DATA7       PIN_B7
#include <lcd.c>

#define d2 200  // Delay para la animación

int i = 0;
int contador = 0;

// Variables de tiempo
int hd = 2;
int hu = 3;
int md = 5;
int mu = 8;
int sd = 5;
int su = 7;

//Caracteres animados
int pac_man1[8] = {14,27,31,24,28,31,14,0};
int pac_man2[8] = {14,27,31,31,31,31,14,0};
int bola[8] = {0,0,0,6,6,0,0,0};
int bolaG[8] = {0,0,0,14,14,14,0,0};
int pac_man1R[8] = {14,27,31,3,3,31,14,0};
int fantasma[8] = {14,31,21,31,31,31,21,21};
int espacio[8] = {0,0,0,0,0,0,0,0};


void reloj(){
   su++; //Aumentamos un segundo al reloj
   if(su == 10){// aumentamos una decena a los segundos
      su = 0;
      sd++;
   }
   
   if(sd == 6){// aumentamos un minuto
      sd = 0;
      mu++;
   }
   
   if(mu == 10){// aumentamos una decena a los minutos y reiniciamos unidades
      mu = 0;
      md++;
   }
   
   if(md == 6){// aumentamos una hora
      md = 0;
      hu++;
   }
   
   if(hu == 10){// Aumentamos una decena a las horas
      hu=0;
      hd++;
   }
   // Al llegar a 24 horas se reinicia el reloj
   if(hd == 2 && hu == 4){
      su = 0;
      sd = 0;
      mu = 0;
      md = 0;
      hu = 0;
      hd = 0;
   }
   // Reiniciamos el timer
   set_timer0(100);
}

// Rutina de animación y actualización del reloj
void animacion(){
   delay_ms(d2);
   lcd_gotoxy(1,2);
   printf(lcd_putc,"%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c",2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2);
   lcd_gotoxy(1,1);
   printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su);
   delay_ms(d2);
   lcd_gotoxy(1,2);
   printf(lcd_putc,"%c",1);
   lcd_gotoxy(1,1);
   printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su);
   delay_ms(d2);
   lcd_gotoxy(1,2);
   printf(lcd_putc,"%c",0);
   lcd_gotoxy(1,1);
   printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su);
   delay_ms(d2);
   lcd_gotoxy(1,2);
   printf(lcd_putc,"%c%c",5,1);
   lcd_gotoxy(1,1);
   printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su);
   delay_ms(d2);
   lcd_gotoxy(1,2);
   printf(lcd_putc,"%c%c",5,0);
   lcd_gotoxy(1,1);
   printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su);
   delay_ms(d2);

   for(int i=0; i<=15;i++){
      lcd_gotoxy(i,2);
      printf(lcd_putc,"%c%c%c%c",5,4,5,1);   
      lcd_gotoxy(1,1);
      printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su);
      delay_ms(d2);
      lcd_gotoxy(i,2);
      printf(lcd_putc,"%c%c%c%c",5,4,5,0);  
      lcd_gotoxy(1,1);
      printf(lcd_putc,"    %u%u:%u%u:%u%u   ",hd,hu,md,mu,sd,su); 
      delay_ms(d2);
   }
}

//Función de interrupción
#INT_TIMER0
void timer0_isr(void){
   contador++;
   if(contador == 200){
      contador = 0;
      reloj();
   }
}

void main(){
   // Iniciamos lcd
   lcd_init();
   // Declaramos los caracteres especiales en la memoria
   lcd_set_cgram_char(0,pac_man1);
   lcd_set_cgram_char(1,pac_man2);
   lcd_set_cgram_char(2,bola);
   lcd_set_cgram_char(3,pac_man1R);
   lcd_set_cgram_char(4,fantasma);
   lcd_set_cgram_char(5,espacio);
   lcd_set_cgram_char(6,bolaG);
   printf(lcd_putc,"\f");
   lcd_gotoxy(1,1);
   
   //Debido a la carga de la simulación
   //se intentó que el timer considira aunque en la realidad este puede variar
   setup_timer_0(RTCC_INTERNAL|RTCC_DIV_16);
   set_timer0(100);
   enable_interrupts(INT_TIMER0);
   enable_interrupts(GLOBAL);
   
   while(TRUE){
      animacion();
   }
}

