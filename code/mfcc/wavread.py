import wave
import numpy

def gate(inputfile):
    f = wave.open(inputfile,'rb')
    params = f.getparams()
    nchannels,sampwidth,framerate,nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    
    wave_data = numpy.fromstring(str_data,dtype = numpy.short)
    #wave_data = wave_data.T
    maxim = max(wave_data.max(),abs(wave_data.min()))
    wave_data[abs(wave_data[:])<maxim/10] = 0  
    
    f = wave.open(inputfile,'wb')
    f.setnchannels(nchannels)
    f.setsampwidth(sampwidth)
    f.setframerate(framerate)
    f.writeframes(wave_data.tostring())
    f.close

if __name__ == '__main__':
    inputfile = '1.wav'
    gate(inputfile)


