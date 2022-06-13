#include <16f877a.h>
#fuses HS,NOWDT,NOPROTECT,NOPUT,NOLVP,BROWNOUT
#use delay(clock=20M)
#use standard_io(B)
#use standard_io(D)

#define led_red PIN_D0
#define led_green PIN_D1
#define led_blue PIN_D2

#define LCD_DB4   PIN_B4         // Pines de la pantalla LCD
#define LCD_DB5   PIN_B5
#define LCD_DB6   PIN_B6
#define LCD_DB7   PIN_B7
#define LCD_RS    PIN_B2
#define LCD_E     PIN_B3

#define P_TCS230 PIN_D5
#define S2 PIN_D6
#define S3 PIN_D7
#include <TCS230.c>
#include <LCD_16X2.c>

long red = 0;
long green = 0;
long blue = 0;

void main()
{
   lcd_init();
   TCS230_init();
   output_low(led_red);
   output_low(led_green);
   output_low(led_blue);
   
   while(true)
   {
      red = TCS230_getFrequence_red();                  // Lectura para el color rojo
      green = TCS230_getFrequence_green();              // Lectura para el color verde
      blue = TCS230_getFrequence_blue();                // Lectura para el color azul
      
      output_low(led_red);
      output_low(led_green);
      output_low(led_blue);
     
      if(blue > 20 && blue < 70 && green > 58 && green < 88)
      {
         output_high(led_blue);
      }
      
      if(blue > 65 && blue < 98 && green > 90 && green < 120 && red > 0 && red < 60)
      {
         output_high(led_red);
      }
      
      if(green > 35 && green < 55)
      {
         output_high(led_green);
      }
      
      lcd_clear();
      lcd_gotoxy(1,1);
      printf(lcd_putc,"R: %Lu", red);
      lcd_gotoxy(1,2);
      printf(lcd_putc,"G: %Lu", green);
      lcd_gotoxy(9,1);
      printf(lcd_putc,"B: %Lu", blue);
      delay_ms(250);
   }
}
