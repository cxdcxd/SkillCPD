﻿using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class KeyInput_shoulder_lift : MonoBehaviour
{

    public float speed = 30.0f;


    private ArticulationBody articulation;

    // Start is called before the first frame update
    void Start()
    {
        articulation = GetComponent<ArticulationBody>();
    }

    // Update is called once per frame
    void Update()
    {
        float rotationChange;
        float rotationGoal = CurrentPrimaryAxisRotation();
        if (Input.GetKey(KeyCode.D))
        {
            rotationChange =  speed * Time.fixedDeltaTime;
            rotationGoal += rotationChange;
            RotateTo(rotationGoal);
            print("D");
        }
        else if (Input.GetKey(KeyCode.A))
        {
            rotationChange = speed * Time.fixedDeltaTime;
            rotationGoal -= rotationChange;
            RotateTo(rotationGoal);
            print("A");
        }

    }

    float CurrentPrimaryAxisRotation()
    {
        float currentRotationRads = articulation.jointPosition[0];
        float currentRotation = Mathf.Rad2Deg * currentRotationRads;
        return currentRotation;
    }

    void RotateTo(float primaryAxisRotation)
    {
        var drive = articulation.xDrive;
        drive.target = primaryAxisRotation;
        articulation.xDrive = drive;
    }
}
