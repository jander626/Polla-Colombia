"""Bot de señales de trading (acciones + forex) para Quantfury.

El paquete genera señales long-only sobre velas diarias y las entrega por
Telegram. No se conecta a Quantfury: Quantfury actúa como restricción de
universo (solo se recomiendan instrumentos operables allí), y la ejecución
de las órdenes es manual.
"""
