#include <cmath>
#include <iostream>
//#include <stdio.h>
//#include <windows.h>
#include "mpdev.h"

using namespace std;
extern "C" {

int start_api() {
    MPRETURNCODE retval1;
    retval1 = connectMPDev(MP160, MPUDP, "auto");
    if (retval1 != MPSUCCESS) {
        std::cout << "Program failed to connect to MP Device" << endl;
        std::cout << "connectMPDev returned with " << retval1 << " as a return code." << endl;
        std::cout << "Disconnecting..." << endl;
        disconnectMPDev();
        cout << "Exit" << endl;
        return 0;
    }
    MPRETURNCODE retval;
    //acquire on channel 2, 7
    BOOL analogCH[] = {false, true, false, false,
                       false, false, true, false,
                       false, false, false, false,
                       false, false, false, false};

    cout << "Acquire data on  Analog Channel 1 and Analog Channel 5" << endl;
    cout << "Setting Acquisition Channels..." << endl;
    retval = setAcqChannels(analogCH);
    if (retval != MPSUCCESS) {
        cout << "Program failed to set Acquisition Channels" << endl;
        cout << "setAcqChannels returned with " << retval << " as a return code." << endl;
        return 0;
    }

    //set sample rate to 5 msec per sample = 200 Hz
    cout << "Setting Sample Rate to 200 Hz" << endl;
    retval = setSampleRate(5.0);

    if (retval != MPSUCCESS) {
        cout << "Program failed to set Sample Rate" << endl;
        cout << "setSampleRate returned with " << retval << " as a return code." << endl;
        return 0;
    }

    cout << "Starting Acquisition Daemon..." << endl;
    retval = startMPAcqDaemon();

    if (retval != MPSUCCESS) {
        cout << "Program failed to Start Acquisition Daemon" << endl;
        cout << "startMPAcqDaemon returned with " << retval << " as a return code." << endl;
        cout << "Stopping..." << endl;
        stopAcquisition();
        return 0;
    }

    cout << "Daemon Started" << endl;

    cout << "Starting Acquisition..." << endl;
    retval = startAcquisition();

    if (retval != MPSUCCESS) {
        cout << "Program failed to Start Acquisition" << endl;
        cout << "startAcquisition returned with " << retval << " as a return code." << endl;
        cout << "Stopping..." << endl;
        stopAcquisition();
        return 0;
    }
    return 1;
}
void stop_api() {
    cout << "Stopping..." << endl;
    stopAcquisition();
    cout << "Disconnecting..." << endl;
    disconnectMPDev();
    cout << "Exit" << endl;
}

int get_ecg_ppg_data(double* buff, int sec) {
    long numsamples = sec * 200;
    DWORD valuesRead = 0;
    DWORD numValuesToRead = 0;
    //remember that data will be interleaved
    //therefore we need to mulitply the number of samples
    //by the number of active channels to acquire the necessary
    //data points from the active channels
    DWORD remainValues = 2 * numsamples;
    DWORD offset = 0;

//    cout << "Acquiring...\n" << endl;
    while (remainValues > 0) {
        numValuesToRead = 2 * 200;
        numValuesToRead = (numValuesToRead > remainValues) ? remainValues : numValuesToRead;
        if (receiveMPData(buff + offset, numValuesToRead, &valuesRead) != MPSUCCESS) {
            cout << "Failed to receive MP data" << endl;
            char szbuff3[512];
            memset(szbuff3, 0, 512);
            sprintf(szbuff3, "Failed to Recv data from Acq Daemon. Last ERROR=%d, Read=%d", getMPDaemonLastError(),
                    valuesRead);
            cout << szbuff3 << endl;
            stopAcquisition();
            break;
        }
        offset += numValuesToRead;
        remainValues -= numValuesToRead;
    }
    return 0;
}


}