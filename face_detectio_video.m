% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();
eyesDetector = vision.CascadeObjectDetector('EyePairBig');
mouthDetector = vision.CascadeObjectDetector('Mouth');
eyes.Detector.MergeThreshold=7;
mouthDetector.MergeThreshold=120;
% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
pointTrackerEyes = vision.PointTracker('MaxBidirectionalError', 2);
pointTrackerMouth = vision.PointTracker('MaxBidirectionalError', 2);
% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);



runLoop = true;
numPts = 0;
numPtsEyes = 0;
numPtsMouth = 0;
frameCount = 0;

while runLoop

    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 10
        % Detection mode.
        bbox = faceDetector.step(videoFrameGray);
        

        if ~isempty(bbox)
            % Find corner points inside the detected region.
            % uses the minimum eigenvalue algorithm developed by Shi and Tomasi to find feature points.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
           

            % Re-initialize the point tracker.
            xyPoints = points.Location;
            

            numPts = size(xyPoints,1);
            
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
           
            % Save a copy of the points.
            oldPoints = xyPoints;
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(1, :));
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            
            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

           
        end

    else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        
        visiblePoints = xyPoints(isFound, :);
        
        oldInliers = oldPoints(isFound, :);
        
        
        numPts = size(visiblePoints, 1);
       
        
        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);


            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end
    
    
    
    % Eyes Pair
    if numPtsEyes < 10
        % Detection mode.
        bboxEyes = eyesDetector.step(videoFrameGray);
        

        if ~isempty(bboxEyes)
            % Find corner points inside the detected region.
            % uses the minimum eigenvalue algorithm developed by Shi and Tomasi to find feature points.
            pointsEyes = detectMinEigenFeatures(videoFrameGray, 'ROI', bboxEyes(1, :));
           

            % Re-initialize the point tracker.
            xyPointsEyes = pointsEyes.Location;
            

            numPtsEyes = size(xyPointsEyes,1);
            
            release(pointTrackerEyes);
            initialize(pointTrackerEyes, xyPointsEyes, videoFrameGray);
           
            % Save a copy of the points.
            oldPointsEyes = xyPointsEyes;
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPointsEyes = bbox2points(bboxEyes(1, :));
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygonEyes = reshape(bboxPointsEyes', 1, []);
            
            
            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonEyes, 'LineWidth', 3);

           
        end

    else
        % Tracking mode.
        [xyPointsEyes, isFoundEyes] = step(pointTrackerEyes, videoFrameGray);
        
        visiblePointsEyes = xyPointsEyes(isFoundEyes, :);
        
        oldInliersEyes = oldPointsEyes(isFoundEyes, :);
        
        
        numPtsEyes = size(visiblePointsEyes, 1);
       
        
        if numPtsEyes >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xformEyes, oldInliersEyes, visiblePointsEyes] = estimateGeometricTransform(...
                oldInliersEyes, visiblePointsEyes, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box.
            bboxPointsEyes = transformPointsForward(xformEyes, bboxPointsEyes);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygonEyes = reshape(bboxPointsEyes', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonEyes, 'LineWidth', 3);


            % Reset the points.
            oldPointsEyes = visiblePointsEyes;
            setPoints(pointTrackerEyes, oldPointsEyes);
        end

    end
    
    
    
    
    
    
    
    % Mouth
    if numPtsMouth < 90
        % Detection mode
        bboxMouth = mouthDetector.step(videoFrameGray);
        disp(~isempty(bboxMouth));

        if ~isempty(bboxMouth)
            % Find corner points inside the detected region.
            % uses the minimum eigenvalue algorithm developed by Shi and Tomasi to find feature points.
            pointsMouth = detectMinEigenFeatures(videoFrameGray, 'ROI', bboxMouth(1, :));
            disp('coming coming')

            % Re-initialize the point tracker.
            xyPointsMouth = pointsMouth.Location;
            

            numPtsMouth = size(xyPointsMouth,1);
            
            release(pointTrackerMouth);
            initialize(pointTrackerMouth, xyPointsMouth, videoFrameGray);
           
            % Save a copy of the points.
            oldPointsMouth = xyPointsMouth;
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPointsMouth = bbox2points(bboxMouth(1, :));
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygonMouth = reshape(bboxPointsMouth', 1, []);
            
            
            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonMouth, 'LineWidth', 3);

           
        end

    else
        % Tracking mode.
        [xyPointsMouth, isFoundMouth] = step(pointTrackerMouth, videoFrameGray);
        
        visiblePointsMouth = xyPointsMouth(isFoundMouth, :);
        
        oldInliersMouth = oldPointsMouth(isFoundMouth, :);
        
        
        numPtsMouth = size(visiblePointsMouth, 1);
       
        
        if numPtsMouth >= 90
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xformMouth, oldInliersMouth, visiblePointsMouth] = estimateGeometricTransform(...
                oldInliersMouth, visiblePointsMouth, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box.
            bboxPointsMouth = transformPointsForward(xformMouth, bboxPointsMouth);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygonMouth = reshape(bboxPointsMouth', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygonMouth, 'LineWidth', 3);


            % Reset the points.
            oldPointsMouth = visiblePointsMouth;
            setPoints(pointTrackerMouth, oldPointsMouth);
        end

    end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(pointTrackerEyes);
release(pointTrackerMouth);
release(faceDetector);
release(eyesDetector);
release(mouthDetector);