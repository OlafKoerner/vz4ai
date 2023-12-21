#!/bin/bash
sudo systemctl stop middleware.service
sudo systemctl stop vzlogger.service
sleep 2
sudo systemctl start middleware.service
sleep 2
sudo systemctl start vzlogger.service

