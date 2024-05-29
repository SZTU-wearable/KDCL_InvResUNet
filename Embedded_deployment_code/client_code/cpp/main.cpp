#include <iostream>
#include <stdio.h>
#include <windows.h>
#include "mpdev.h"

using namespace std;

void getBufferDemo();

void startAcqDaemonDemo();

void ioDemo();

void get_ecg_ppg_data();

int main(int argc, char **argv) {
    MPRETURNCODE retval;
    //configure the API and connect to the MP Device
    //Note: Currently set to automatically discover the MP150.
    //Change the third parameter the serial number of the MP150
    //if it is not the BHAPI is not connecting to the correct MP150
    retval = connectMPDev(MP160, MPUDP, "auto");

    if (retval != MPSUCCESS) {
        cout << "Program failed to connect to MP Device" << endl;
        cout << "connectMPDev returned with " << retval << " as a return code." << endl;
        cout << "Disconnecting..." << endl;
        disconnectMPDev();
        cout << "Exit" << endl;
        return 0;
    }

    //execute Start Acquistion Daemon Demo
    cout << "Executing Start Acquistion Daemon Demo..." << endl;
    startAcqDaemonDemo();
//    get_ecg_ppg_data();
    disconnectMPDev();

    cout << "Exit" << endl;

    return 1;
}

void get_ecg_ppg_data() {
    MPRETURNCODE retval;
    //acquire on channel 2, 7, and 11
    BOOL analogCH[] = {false, true, false, false,
                       false, false, true, false,
                       false, false, true, false,
                       false, false, false, false};

    cout << "Acquire data on  Analog Channel 1 and Analog Channel 5" << endl;
    cout << "Setting Acquisition Channels..." << endl;
    retval = setAcqChannels(analogCH);

    if (retval != MPSUCCESS) {
        cout << "Program failed to set Acquisition Channels" << endl;
        cout << "setAcqChannels returned with " << retval << " as a return code." << endl;
        return;
    }

    //set sample rate to 5 msec per sample = 200 Hz
    cout << "Setting Sample Rate to 200 Hz" << endl;
    retval = setSampleRate(5.0);

    if (retval != MPSUCCESS) {
        cout << "Program failed to set Sample Rate" << endl;
        cout << "setSampleRate returned with " << retval << " as a return code." << endl;
        return;
    }

    cout << "Starting Acquisition Daemon..." << endl;
    retval = startMPAcqDaemon();
    if (retval != MPSUCCESS) {
        cout << "Program failed to Start Acquisition Daemon" << endl;
        cout << "startMPAcqDaemon returned with " << retval << " as a return code." << endl;
        cout << "Stopping..." << endl;
        stopAcquisition();
        return;
    }

    cout << "Daemon Started" << endl;
    cout << "Starting Acquisition..." << endl;
    retval = startAcquisition();

    if (retval != MPSUCCESS) {
        cout << "Program failed to Start Acquisition" << endl;
        cout << "startAcquisition returned with " << retval << " as a return code." << endl;
        cout << "Stopping..." << endl;
        stopAcquisition();
        return;
    }

    cout << "Acquiring..." << endl;
    long numsamples = 1000;
    cout << "Getting " << numsamples << " samples from the MP Device... " << endl;

    DWORD valuesRead = 0;
    DWORD numValuesToRead = 0;
    //remember that data will be interleaved
    //therefore we need to mulitply the number of samples
    //by the number of active channels to acquire the necessary
    //data points from the active channels
    DWORD remainValues = 3 * numsamples;
    double *buff = new double[remainValues];
    DWORD offset = 0;

    cout << "Acquiring...\n" << endl;

    while (remainValues > 0) {
        //read 1 second of data at a time
        //frequency times the number of active channels
        numValuesToRead = 3 * 200;

        //if there are more values to be read than the remaing number of values we want read then just read the reamining needed
        numValuesToRead = (numValuesToRead > remainValues) ? remainValues : numValuesToRead;

        if (receiveMPData(buff + offset, numValuesToRead, &valuesRead) != MPSUCCESS) {
            cout << "Failed to receive MP data" << endl;
            // using of getMPDaemonLAstError is a good practice
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
        //show status
        printf("                                                      \r");
        printf("Remaining Values: %d\r", remainValues);
    }

    cout << endl;

    for (int j = 0; j < numsamples; j++) {
        cout << "Sample: " << j + 1 << endl;

        for (int i = 0; i < 3; i++)
            switch (i) {
                case 0:
                    cout << "CH2: " << buff[i + (3 * j)];
                    break;
                case 1:
                    cout << " CH7: " << buff[i + (3 * j)];
                    break;
                case 2:
                    cout << " CH11: " << buff[i + (3 * j)] << endl;
                    break;
                default:
                    break;

            }
    }
    //stop
    cout << "Stopping..." << endl;

    stopAcquisition();

    //free Memory
    delete[] buff;
}

// Start Acquistion Daemon Demo
void startAcqDaemonDemo() {
    MPRETURNCODE retval;

    //acquire on channel 2, 7, and 11
    BOOL analogCH[] = {false, true, false, false,
                       false, false, true, false,
                       false, false, true, false,
                       false, false, false, false};

    cout << "Acquire data on  Analog Channel 1 and Analog Channel 5" << endl;
    cout << "Setting Acquisition Channels..." << endl;
    retval = setAcqChannels(analogCH);

    if (retval != MPSUCCESS) {
        cout << "Program failed to set Acquisition Channels" << endl;
        cout << "setAcqChannels returned with " << retval << " as a return code." << endl;

        return;
    }

    //set sample rate to 5 msec per sample = 200 Hz
    cout << "Setting Sample Rate to 200 Hz" << endl;
    retval = setSampleRate(5.0);

    if (retval != MPSUCCESS) {
        cout << "Program failed to set Sample Rate" << endl;
        cout << "setSampleRate returned with " << retval << " as a return code." << endl;

        return;
    }

    cout << "Starting Acquisition Daemon..." << endl;
    retval = startMPAcqDaemon();

    if (retval != MPSUCCESS) {
        cout << "Program failed to Start Acquisition Daemon" << endl;
        cout << "startMPAcqDaemon returned with " << retval << " as a return code." << endl;

        cout << "Stopping..." << endl;

        stopAcquisition();

        return;
    }

    cout << "Daemon Started" << endl;

    cout << "Starting Acquisition..." << endl;
    retval = startAcquisition();

    if (retval != MPSUCCESS) {
        cout << "Program failed to Start Acquisition" << endl;
        cout << "startAcquisition returned with " << retval << " as a return code." << endl;

        cout << "Stopping..." << endl;

        stopAcquisition();

        return;
    }

    cout << "Acquiring..." << endl;

    long numsamples = 10000;

    cout << "Getting " << numsamples << " samples from the MP Device... " << endl;

    DWORD valuesRead = 0;
    DWORD numValuesToRead = 0;
    //remember that data will be interleaved
    //therefore we need to mulitply the number of samples
    //by the number of active channels to acquire the necessary
    //data points from the active channels
    DWORD remainValues = 3 * numsamples;
    double *buff = new double[remainValues];
    DWORD offset = 0;

    cout << "Acquiring...\n" << endl;

    while (remainValues > 0) {
        //read 1 second of data at a time
        //frequency times the number of active channels
        numValuesToRead = 3 * 200;

        //if there are more values to be read than the remaing number of values we want read then just read the reamining needed
        numValuesToRead = (numValuesToRead > remainValues) ? remainValues : numValuesToRead;

        if (receiveMPData(buff + offset, numValuesToRead, &valuesRead) != MPSUCCESS) {
            cout << "Failed to receive MP data" << endl;


            // using of getMPDaemonLAstError is a good practice
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

        //show status
        printf("                                                      \r");
        printf("Remaining Values: %d\r", remainValues);
    }

    cout << endl;

    for (int j = 0; j < numsamples; j++) {
        cout << "Sample: " << j + 1 << endl;

        for (int i = 0; i < 3; i++)
            switch (i) {
                case 0:
                    cout << "CH2: " << buff[i + (3 * j)];
                    break;
                case 1:
                    cout << " CH7: " << buff[i + (3 * j)];
                    break;
                case 2:
                    cout << " CH11: " << buff[i + (3 * j)] << endl;
                    break;
                default:
                    break;

            }
    }

    //stop
    cout << "Stopping..." << endl;

    stopAcquisition();

    //free Memory
    delete[] buff;
}
