
#!/bin/sh

echo "Setting SVE VL to 2048 bits"
echo 256 > /proc/sys/abi/sve_default_vector_length

echo "Starting the sve-test..."
./sve-test
