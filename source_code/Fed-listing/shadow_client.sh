for cid in {0..9}; do
   python shadow_client.py --cid $cid &   # change file name for each setting, 
                                          # FOr random setting: use file (shadow_cient_random.py)
                                          # For equal distribution: use (shadow_client_equal_dist.py)
                                          # FOr one class missing: use (shadow_client_class_missing.py)
                                          # FOr single class only: use (shadow_client_single_class.py)
done
wait
