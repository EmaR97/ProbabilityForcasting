import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


# Definisci la funzione
def funzione(x, y, a, b):
    return xa * np.exp(-b * (y - np.sin(x)) ** 2)


# Crea un set di dati per x e y
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 20, 1000)
x, y = np.meshgrid(x, y)

# Imposta i parametri a e b
a = 1
b = 0.1

# Calcola i valori della funzione
z = funzione(x, y, a, b)

# Visualizza il grafico 3D con dimensioni più grandi e un'inclinazione maggiore
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# Aggiungi etichette
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Imposta l'inclinazione
ax.view_init(elev=30, azim=45)  # Modifica i valori di elev e azim secondo le tue preferenze

# Mostra il grafico
plt.show()

# Define the integration limits
x_lower, x_upper = 0, 10
y_lower, y_upper = -np.inf, np.inf


z_function = lambda y, x: a * np.exp(-b * (y - np.sin(x)) ** 2)

z = z_function(y, x)

res_y_more_tan = integrate.dblquad(z_function, x_lower, x_upper, 5, np.inf)
print(res_y_more_tan)
res_tot = integrate.dblquad(z_function, x_lower, x_upper, -np.inf, np.inf)
print(res_tot)
print((res_y_more_tan[0] / res_tot[0]) * 100)
