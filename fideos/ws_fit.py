import echelle_orders
import fideos_spectrograph

rv = 0
o = 70 #63, 104
order = echelle_orders.init_stellar_doppler_simple(rv, o)
spec = fideos_spectrograph.tracing(order)

print(spec)
