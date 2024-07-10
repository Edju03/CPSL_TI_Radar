#include "DCA1000Handler.hpp"

/**
 * @brief Construct a new DCA1000Handler::DCA1000Handler object
 * 
 * @param configReader 
 */
DCA1000Handler::DCA1000Handler( const SystemConfigReader& configReader,
                                const RadarConfigReader& radarConfigReader):
    system_config_reader(),
    radar_config_reader(),
    initialized(false),
    DCA_fpgaIP(""),
    DCA_systemIP(""),
    DCA_cmdPort(-1),
    DCA_dataPort(-1),
    save_to_file(false),
    cmd_socket(-1),
    data_socket(-1),
    dropped_packets(0),
    dropped_packet_events(0),
    received_packets(0),
    adc_data_byte_count(0),
    received_frames(0),
    next_frame_byte_buffer_idx(0)
    {     
        initialize(configReader,radarConfigReader);
    }

/**
 * @brief Destroy the DCA1000Handler::DCA1000Handler object
 * 
 */
DCA1000Handler::~DCA1000Handler() {
    if (cmd_socket >= 0) {
        close(cmd_socket);
    }
    if (data_socket >= 0){
        close(data_socket);
    }
    if(save_to_file){
        if(out_file.is_open()){
            out_file.close();
        }
    }
}

bool DCA1000Handler::initialize(
    const SystemConfigReader& systemConfigReader,
    const RadarConfigReader& radarConfigReader){

    initialized = false;

    //load the radar config reader
    radar_config_reader = radarConfigReader;
    if(radar_config_reader.initialized == false){
        return false;
    }else{
        init_buffers();
    }

    //load the system configuration information
    system_config_reader = systemConfigReader;
    if(system_config_reader.initialized == false){
        return false;
    } else{
        load_config();
    }

    //initialize file streaming
    if(save_to_file){
        if(init_out_file() != true){
            return false;
        }
    }

    //initialize the addresses
    init_addresses();

    //initialize sockets
    if(init_sockets() != true){
        return false;
    }

    //set initialization status to true
    initialized = true;

    //configure the DCA1000
    if(configure_DCA1000() != true){
        initialized = false; //initializing the DCA1000 falied
        return false;
    }
}

/**
 * @brief Load required information from the system_config_reader
 * 
 */
void DCA1000Handler::load_config(){

    DCA_fpgaIP = system_config_reader.getDCAFpgaIP();
    DCA_systemIP = system_config_reader.getDCASystemIP();
    DCA_cmdPort = system_config_reader.getDCACmdPort();
    DCA_dataPort = system_config_reader.getDCADataPort();
    save_to_file = system_config_reader.get_save_to_file();

    //print key ports
    std::cout << "FPGA IP: " << DCA_fpgaIP << std::endl;
    std::cout << "System IP: " << DCA_systemIP << std::endl;
    std::cout << "cmd port: " << DCA_cmdPort << std::endl;
    std::cout << "data port: " << DCA_dataPort << std::endl;
}

/**
 * @brief Setup the command and data addresses
 * 
 */
void DCA1000Handler::init_addresses() {
    //setup config address
    cmd_address.sin_family = AF_INET;
    cmd_address.sin_addr.s_addr = inet_addr(DCA_systemIP.c_str());
    cmd_address.sin_port = htons(DCA_cmdPort);

    //setup the data address
    data_address.sin_family = AF_INET;
    data_address.sin_addr.s_addr = inet_addr(DCA_systemIP.c_str());
    data_address.sin_port = htons(DCA_dataPort);
}

/**
 * @brief 
 * 
 * @return true 
 * @return false 
 */
bool DCA1000Handler::init_sockets() {

    // Create socket
    cmd_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (cmd_socket < 0) {
        std::cerr << "Failed to create cmd socket" << std::endl;
        return false;
    }

    data_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (data_socket < 0) {
        std::cerr << "Failed to create data socket" << std::endl;
        return false;
    }

    // Set socket timeout
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(cmd_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(data_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    //bind to command socket
    if (bind(cmd_socket, (struct sockaddr*)&cmd_address, sizeof(cmd_address)) < 0) {
        std::cerr << "Failed to bind socket" << std::endl;
        close(cmd_socket);
        cmd_socket = -1;
        return false;
    } else{
        std::cout << "Bound to command socket" <<std::endl;
    }

    //bind to data socket
    if (bind(data_socket, (struct sockaddr*)&data_address, sizeof(data_address)) < 0) {
        std::cerr << "Failed to bind socket" << std::endl;
        close(data_socket);
        data_socket = -1;
        return false;
    } else{
        std::cout << "Bound to data socket" <<std::endl;
    }

    return true;
}

/**
 * @brief Send a series of commands to the DCA1000 to configure it
 * 
 * @return true - DCA1000 successfully configured
 * @return false - DCA1000 not successfully configured
 */
bool DCA1000Handler::configure_DCA1000(){

    if (initialized){
        //send system connect
        if(send_systemConnect() != true){
            return false;
        }

        //send reset FPGA
        if(send_resetFPGA() != true){
            return false;
        }

        //send configure packet data
        if(send_configPacketData(1472,25) != true){
            return false;
        }

        //send config FPGA gen
        if(send_configFPGAGen() != true){
            return false;
        }

        //read the FPGA version
        float fpga_version = send_readFPGAVersion();

        if(fpga_version > 0){
            std::cout << "FPGA (firmware version: " << fpga_version << ") initialized successfully" << std::endl;
            return true;
        } else{
            return false;
        }
    } else{
        std::cerr << "attempted to configure DCA1000, but DCA1000 Handler wasn't initialized";
        return false;
    }
}

/**
 * @brief 
 * 
 * @param command 
 * @return true 
 * @return false 
 */
bool DCA1000Handler::sendCommand(std::vector<uint8_t>& command) {
    if (cmd_socket < 0) {
        std::cerr << "Socket not bound" << std::endl;
        return false;
    }

    //define address to send to
    struct sockaddr_in fpgaAddr;
    fpgaAddr.sin_family = AF_INET;
    fpgaAddr.sin_addr.s_addr = inet_addr(DCA_fpgaIP.c_str());
    fpgaAddr.sin_port = htons(DCA_cmdPort);

    ssize_t sentBytes = sendto(cmd_socket, command.data(), command.size(), 0,
                               (struct sockaddr*)&fpgaAddr, sizeof(fpgaAddr));
    if (sentBytes != static_cast<ssize_t>(command.size())) {
        std::cerr << "Failed to send command" << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief 
 * 
 * @param buffer 
 * @return true 
 * @return false 
 */
bool DCA1000Handler::receiveResponse(std::vector<uint8_t>& buffer) {
    if (cmd_socket < 0) {
        std::cerr << "Socket not bound" << std::endl;
        return false;
    }

    struct sockaddr_in fromAddr;
    socklen_t fromLen = sizeof(fromAddr);
    ssize_t receivedBytes = recvfrom(cmd_socket, buffer.data(), buffer.size(), 0,
                             (struct sockaddr*)&fromAddr, &fromLen);
    if (receivedBytes < 0) {
        std::cerr << "Failed to receive data" << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief 
 * 
 * @return true 
 * @return false 
 */
bool DCA1000Handler::send_resetFPGA(){

    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                        DCA1000Commands::RESET_FPGA);
    
    //send command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){

        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //confirm success
        if (status == 0){
            return true;
        }else{
            return false;
        }
    } else{
        return false;
    }
}

bool DCA1000Handler::send_recordStart(){
    
    //get the command
    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                        DCA1000Commands::RECORD_START);
    
    //send command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){

        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //confirm success
        if (status == 0){
            return true;
        }else{
            return false;
        }
    } else{
        return false;
    }
}

bool DCA1000Handler::send_recordStop(){
    
    //get the command
    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                        DCA1000Commands::RECORD_STOP);
    
    //send command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){

        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //confirm success
        if (status == 0){
            return true;
        }else{
            return false;
        }
    } else{
        return false;
    }
}

bool DCA1000Handler::send_systemConnect(){
    
    //get the command
    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                        DCA1000Commands::SYSTEM_CONNECT);
    
    //send command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){

        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //confirm success
        if (status == 0){
            return true;
        }else{
            return false;
        }
    } else{
        return false;
    }
}

/**
 * @brief 
 * 
 * @param packet_size 
 * @param delay_us 
 * @return true 
 * @return false 
 */
bool DCA1000Handler::send_configPacketData(size_t packet_size, uint16_t delay_us){

    //set the udp_packet_size
    udp_packet_size = packet_size;

    //declare data vector
    std::vector<uint8_t> data(6,0);

    //define packet size
    std::uint16_t pkt_size = static_cast<std::uint16_t>(packet_size);
    data[0] = static_cast<uint8_t>(pkt_size & 0xFF);
    data[1] = static_cast<uint8_t>((pkt_size >> 8) & 0xFF);

    //define delay
    data[2] = static_cast<uint8_t>(delay_us & 0xFF);
    data[3] = static_cast<uint8_t>((delay_us >> 8) & 0xFF);

    // bytes 4 & 5 are future use

    //generate the command
    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                        DCA1000Commands::CONFIG_PACKET_DATA,
                                        data);    

    //send command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){

        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //confirm success
        if (status == 0){
            return true;
        }else{
            return false;
        }
    } else {
        return false;
    }
}

bool DCA1000Handler::send_configFPGAGen(){
    std::vector<uint8_t> data(6,0);

    //data logging mode - Raw Mode
    data[0] = 0x01;

    //LVDS mode - 4 lane
    data[1] = 0x01;

    //data transfer mode - LVDS capture
    data[2] = 0x01;

    //data capture mode
    data[3] = 0x02;

    //data format mode
    data[4] = 0x03;

    //timer - default to 30 seconds
    data[5] = 30;

    //generate the command
    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                        DCA1000Commands::CONFIG_FPGA_GEN,
                                        data);
    
    //send command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){

        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //confirm success
        if (status == 0){
            return true;
        }else{
            return false;
        }
    } else {
        return false;
    }
}

/**
 * @brief 
 * 
 * @return float 
 */
float DCA1000Handler::send_readFPGAVersion(){

    //get the command
    std::vector<uint8_t> cmd = DCA1000Commands::construct_command(
                                    DCA1000Commands::READ_FPGA_VERSION);

    //send the command
    sendCommand(cmd);

    //get the response
    std::vector<uint8_t> rcv_data(8,0);
    if (receiveResponse(rcv_data)){
        
        //get the status
        uint16_t status = static_cast<uint16_t>(rcv_data[5]) << 8;
        status = status | static_cast<uint16_t>(rcv_data[4]);

        //get version numbers
        uint16_t major_version = (status & 0b01111111);
        uint16_t minor_version = (status >> 7) & 0b01111111;

        return static_cast<float>(major_version) + (static_cast<float>(minor_version)*1e-1);
    }else{
        return 0.0;
    }
}


void DCA1000Handler::init_buffers()
{
    if(radar_config_reader.initialized){
        bytes_per_frame = radar_config_reader.get_bytes_per_frame();
        samples_per_chirp = radar_config_reader.get_samples_per_chirp();
        chirps_per_frame = radar_config_reader.get_chirps_per_frame();
        num_rx_channels = radar_config_reader.get_num_rx_antennas();

        //configure the udp packet buffer
        udp_packet_buffer = std::vector<uint8_t>(udp_packet_size,0);

        //configure the frame byte buffer (assembly)
        frame_byte_buffer = std::vector<uint8_t>(bytes_per_frame,0);
        next_frame_byte_buffer_idx = 0;

        //configure processing of completed frames
        latest_frame_byte_buffer = std::vector<uint8_t>(bytes_per_frame,0);
        new_frame_available = false;

        //reset the dropped packet counting
        received_packets = 0;
        adc_data_byte_count = 0;
        dropped_packets = 0;
        received_frames = 0;

        //adc_cube buffer
        //NOTE: indexed by [Rx channel, sample, chirp]
        size_t rx_channels = 4;
        adc_data_cube = std::vector<std::vector<std::vector<std::complex<std::int16_t>>>>(
            num_rx_channels,std::vector<std::vector<std::complex<std::int16_t>>>(
                samples_per_chirp, std::vector<std::complex<std::int16_t>>(
                    chirps_per_frame,std::complex<std::int16_t>(0,0)
                )
            )
        );
    }else{
        std::cerr << "attempted to initialize DCA1000 Handler buffers,\
        but radar_config_reader wasn't initialized";
    }
}

/**
 * @brief 
 * 
 * @return true 
 * @return false 
 */
bool DCA1000Handler::process_next_packet(){

    ssize_t received_bytes = get_next_udp_packets(udp_packet_buffer);
    if(received_bytes > 0){

        //get the sequence number
        std::uint32_t packet_sequence_number = get_packet_sequence_number(udp_packet_buffer);

        //determine bytes in the new packet
        std::uint64_t packet_byte_count = get_packet_byte_count(udp_packet_buffer);
        std::uint64_t adc_data_bytes_in_packet = static_cast<std::uint64_t>(received_bytes) - 10;
        
        //check for and handle dropped packets
        if (packet_sequence_number != received_packets + 1){
            std::cout << "d-P: " << packet_sequence_number << std::endl;

            //determine the number of dropped packets
            dropped_packets += (packet_sequence_number - received_packets + 1);
            dropped_packet_events += 1;

            zero_pad_frame_byte_buffer(packet_byte_count);

            //update the received packet total
            received_packets = packet_sequence_number;
        } else{
            received_packets += 1;
        }

        //check to make sure all bytes are accounted for
        if (adc_data_byte_count != packet_byte_count){
            std::cout << "d-B" << std::endl;
            
            zero_pad_frame_byte_buffer(packet_byte_count);

        } else{
            adc_data_byte_count += adc_data_bytes_in_packet;
        }

        //copy the bytes into the frame byte buffer
        for (size_t i = 0; i < adc_data_bytes_in_packet; i++)
        {
            frame_byte_buffer[next_frame_byte_buffer_idx] = udp_packet_buffer[i + 10];
            
            //increment the frame byte buffer index
            next_frame_byte_buffer_idx += 1;
            if(next_frame_byte_buffer_idx == bytes_per_frame){

                //save the completed frame byte buffer and reset it
                save_frame_byte_buffer();
            }
        }
        return true;
    } else{
        return false;
    }
}

/**
 * @brief 
 * 
 * @param buffer 
 * @return true 
 * @return false 
 */
ssize_t DCA1000Handler::get_next_udp_packets(std::vector<uint8_t>& buffer) {
    if (data_socket < 0) {
        std::cerr << "data socket not bound" << std::endl;
        return 0;
    }

    struct sockaddr_in fromAddr;
    socklen_t fromLen = sizeof(fromAddr);
    ssize_t receivedBytes = recvfrom(data_socket, buffer.data(), buffer.size(), 0,
                             (struct sockaddr*)&fromAddr, &fromLen);
    if (receivedBytes < 0) {
        std::cerr << "Failed to receive data" << std::endl;
        return 0;
    } else{
        return receivedBytes;
    }
}

void DCA1000Handler::print_status(){
    std::cout <<
        "frame: " << received_frames << std::endl <<
        "\tpackets: " << received_packets << std::endl <<
        "\tdata bytes: " << adc_data_byte_count << std::endl <<
        "\tdropped packets: " << dropped_packets << std::endl <<
        "\tdropped packet events: " << dropped_packet_events << std::endl;
}


uint32_t DCA1000Handler::get_packet_sequence_number(std::vector<uint8_t>& buffer){
    //get the sequence number
    std::uint32_t packet_sequence_number = 
        static_cast<uint32_t>(buffer[3]) << 24 |
        static_cast<uint32_t>(buffer[2]) << 16 |
        static_cast<uint32_t>(buffer[1]) << 8 |
        static_cast<uint32_t>(buffer[0]);
    packet_sequence_number = le32toh(packet_sequence_number);

    return packet_sequence_number;
}

uint64_t DCA1000Handler::get_packet_byte_count(std::vector<uint8_t>& buffer){
    
    //get the byte count
    uint64_t packet_byte_count = 
            static_cast<uint64_t>(buffer[9]) << 40 |
            static_cast<uint64_t>(buffer[8]) << 32 |
            static_cast<uint64_t>(buffer[7]) << 24 |
            static_cast<uint64_t>(buffer[6]) << 16 |
            static_cast<uint64_t>(buffer[5]) << 8  |
            static_cast<uint64_t>(buffer[4]);
        packet_byte_count = le64toh(packet_byte_count);

    return packet_byte_count;
}

/**
 * @brief 
 * @note Assumes that the frame_byte_buffer is initialized with zeros for each new frame
 * 
 * @param packet_byte_count the byte count from the most recently received packets
 * (i.e. the number of bytes that were supposed to have been received prior to the current
 * packet)
 */
void DCA1000Handler::zero_pad_frame_byte_buffer(std::uint64_t packet_byte_count){

    //determin the number of bytes to fill in
    std::uint64_t bytes_to_fill = packet_byte_count - adc_data_byte_count;

    //make sure we won't overflow the frame byte buffer
    std::uint64_t bytes_remaining = bytes_per_frame - next_frame_byte_buffer_idx;

    //room in the buffer
    if(bytes_remaining > bytes_to_fill){
        next_frame_byte_buffer_idx += bytes_to_fill;
    }else
    {
        //reset the frame byte buffer
        save_frame_byte_buffer();

        //reset the index
        if(bytes_remaining != bytes_to_fill){
            (next_frame_byte_buffer_idx + bytes_to_fill) % bytes_per_frame;
        }
    }
    
    //update the received byte total
    adc_data_byte_count += bytes_to_fill;

}

/**
 * @brief saves the latest frame byte buffer into the 
 * latest_frame_byte_buffer variable, resets the frame_byte_buffer
 * and next_frame_byte_buffer_idx varialbes, and sets the
 * new_frame_available variable to true
 * 
 * @param print_system_status on True, prints status
 * 
 */
void DCA1000Handler::save_frame_byte_buffer(bool print_system_status){

    //copy the frame byte buffer into the latest frame byte buffer
    latest_frame_byte_buffer = frame_byte_buffer;

    //reset the frame byte buffer
    frame_byte_buffer = std::vector<uint8_t>(bytes_per_frame,0);

    //rest the next frame byte buffer idex
    next_frame_byte_buffer_idx = 0;

    //specify that a new frame is available
    new_frame_available = true;

    //increment the frame tracking
    received_frames += 1;

    update_latest_adc_cube_1443();

    if(print_system_status){
        print_status();
    }

    if(save_to_file){
        write_adc_data_cube_to_file();
        // write_vector_to_file(latest_frame_byte_buffer);
    }
}

std::vector<std::int16_t> DCA1000Handler::convert_from_bytes_to_ints(
    std::vector<uint8_t>& in_vector)
{
    std::vector<std::int16_t> out_vector(in_vector.size() / 2,0);
    for (size_t i = 0; i < in_vector.size()/2; i++)
    {
        out_vector[i] = (
            (latest_frame_byte_buffer[i * 2]) | 
            (latest_frame_byte_buffer[i * 2 + 1] << 8)
        );

        //TODO: add in either le16toh() or be16toh() to preserve compatibility
    }
    return out_vector;
}

/**
 * @brief Re-shapes a 1D vector into 2D vector, filling in the cols first
 * 
 * @param in_vector 
 * @param num_rows 
 * @return std::vector<std::vector<std::int16_t>> 
 */
std::vector<std::vector<std::int16_t>> DCA1000Handler::reshape_to_2D(
    std::vector<std::int16_t>& in_vector,
    size_t num_rows)
{
    std::vector<std::vector<std::int16_t>> out_vector(
        num_rows, std::vector<std::int16_t>(
            in_vector.size() / num_rows,0
        )
    );

    size_t in_vector_idx = 0;
    size_t row_idx = 0;
    size_t col_idx = 0;

    while (in_vector_idx < in_vector.size())
    {
        out_vector[row_idx][col_idx] = in_vector[in_vector_idx];

        row_idx += 1;

        if(row_idx >= num_rows){
            row_idx = 0;
            col_idx += 1;
        }

        in_vector_idx += 1;
    }
    
    return out_vector;
}

void DCA1000Handler::update_latest_adc_cube_1443(void)
{   
    std::vector<std::int16_t> adc_data_ints = convert_from_bytes_to_ints(latest_frame_byte_buffer);
        
    //reshape it into lvds lanes [Rx1-4 real, Rx1-4 complex]
    std::vector<std::vector<std::int16_t>> adc_data_reshaped = reshape_to_2D(
        adc_data_ints,num_rx_channels * 2
    );

    //update the adc data cube   
    for (size_t chirp_idx = 0; chirp_idx < chirps_per_frame; chirp_idx++)
    {
        for (size_t sample_idx = 0; sample_idx < samples_per_chirp; sample_idx++)
        {   
            size_t idx = (chirp_idx * samples_per_chirp + sample_idx);
            //determine the index in the 2D reshaped buffer
            for (size_t rx_idx = 0; rx_idx < num_rx_channels; rx_idx++)
            {
                //set the real value
                adc_data_cube[rx_idx][sample_idx][chirp_idx].real(
                    adc_data_reshaped[rx_idx][idx]
                );

                //set the imaginary value
                adc_data_cube[rx_idx][sample_idx][chirp_idx].imag(
                    adc_data_reshaped[rx_idx + num_rx_channels][idx]
                );
            }
            
        }
        
    }
}

std::vector<std::vector<std::vector<std::complex<std::int16_t>>>> DCA1000Handler::get_latest_adc_data_cube(void){
    return adc_data_cube;
}

bool DCA1000Handler::init_out_file(){

    out_file = std::ofstream("adc_data.bin", 
        std::ios::out | std::ofstream::binary | std::ios::trunc);

    if(out_file.is_open() != true){
        std::cout << "Failed to open or create adc_data.bin file" << std::endl;
        return false;
    }

    return true;
}

void DCA1000Handler::write_adc_data_cube_to_file(void){
    
    //initialize real and complex values
    std::int16_t real = 0;
    std::int16_t imag = 0;

    //make sure that the out_file is open
    if(out_file.is_open()){
        for(size_t rx_idx=0; rx_idx < num_rx_channels; rx_idx++){
            for(size_t sample_idx = 0; sample_idx < samples_per_chirp; sample_idx++){
                for(size_t chirp_idx = 0; chirp_idx < chirps_per_frame; chirp_idx++){

                    //write the real part
                    real = adc_data_cube[rx_idx][sample_idx][chirp_idx].real();
                    out_file.write(
                        reinterpret_cast<const char*>(
                            &real),
                        sizeof(real)
                    );

                    //write the imag part
                    imag = adc_data_cube[rx_idx][sample_idx][chirp_idx].imag();
                    out_file.write(
                        reinterpret_cast<const char*>(
                            &imag),
                        sizeof(imag)
                    );
                }
            }
        }
    }else{
        std::cerr << "out_file.bin is not open, failed to save ADC data" <<std::endl;
    }
}

void DCA1000Handler::write_vector_to_file(std::vector<std::int16_t> &vector){
    
    //make sure that the out_file is open
    if(out_file.is_open()){
        for(size_t idx = 0; idx < vector.size(); idx++){

            //write the real part
            out_file.write(
                reinterpret_cast<const char*>(
                    &vector[idx]),
                sizeof(vector[idx])
            );
        }
    }else{
        std::cerr << "out_file.bin is not open, failed to save ADC data" <<std::endl;
    }
}