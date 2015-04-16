import pywt
import math

# Wavelets decomposition functions
def haar_dwt_1(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[0][0]

def haar_dwt_2(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[1][0]

def haar_dwt_3(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[2][0]

def haar_dwt_4(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[2][1]

def haar_dwt_5(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[3][0]

def haar_dwt_6(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[3][1]

def haar_dwt_7(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[3][2]

def haar_dwt_8(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[3][3]

def db_dwt_1(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[0][0]

def db_dwt_2(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[1][0]

def db_dwt_3(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[2][0]

def db_dwt_4(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[2][1]

def db_dwt_5(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[3][0]

def db_dwt_6(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[3][1]

def db_dwt_7(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[3][2]

def db_dwt_8(a):
    wavelets = pywt.wavedec(a,'db1')
    return wavelets[3][3]

def wavelets_average_energy(a,lvl):
    wavelets = pywt.wavedec(a,'haar',level=lvl)
    return math.sqrt(wavelets[0][0] ** 2 + wavelets[0][1] ** 2)

def wavelets_details_energy(a,lvl):
    wavelets = pywt.wavedec(a,'haar',level=lvl)
    return math.sqrt(wavelets[1][0] ** 2 + wavelets[1][1] ** 2)

# Wavelets decomposition functions
def test_dwt_1(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[0][0]

def test_dwt_2(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[0][1]

def test_dwt_3(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[1][0]

def test_dwt_4(a):
    wavelets = pywt.wavedec(a,'haar')
    return wavelets[1][1]

def db_wavelets_average_energy(a,lvl):
    wavelets = pywt.wavedec(a,'db2',level=lvl)
    return math.sqrt(wavelets[0][0] ** 2 + wavelets[0][1] ** 2)

def db_wavelets_details_energy(a,lvl):
    wavelets = pywt.wavedec(a,'db2',level=lvl)
    return math.sqrt(wavelets[1][0] ** 2 + wavelets[1][1] ** 2)

def sym_wavelets_average_energy(a,lvl):
    wavelets = pywt.wavedec(a,'sym2',level=lvl)
    return math.sqrt(wavelets[0][0] ** 2 + wavelets[0][1] ** 2)

def sym_wavelets_details_energy(a,lvl):
    wavelets = pywt.wavedec(a,'sym2',level=lvl)
    return math.sqrt(wavelets[1][0] ** 2 + wavelets[1][1] ** 2)

def coif_wavelets_average_energy(a,lvl):
    wavelets = pywt.wavedec(a,'coif1',level=lvl)
    return math.sqrt(wavelets[0][0] ** 2 + wavelets[0][1] ** 2)

def coif_wavelets_details_energy(a,lvl):
    wavelets = pywt.wavedec(a,'coif1',level=lvl)
    return math.sqrt(wavelets[1][0] ** 2 + wavelets[1][1] ** 2)

def bior_wavelets_average_energy(a,lvl):
    wavelets = pywt.wavedec(a,'coif1',level=lvl)
    return math.sqrt(wavelets[0][0] ** 2 + wavelets[0][1] ** 2)

def bior_wavelets_details_energy(a,lvl):
    wavelets = pywt.wavedec(a,'bior1.1',level=lvl)
    return math.sqrt(wavelets[1][0] ** 2 + wavelets[1][1] ** 2)
