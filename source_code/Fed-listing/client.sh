for i in {0..9}; do
    python client.py --client-id $i &  # Use this file for FL training
done