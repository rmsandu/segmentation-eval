void FindCommonBoundingBox(egItkUC3DImagePtr &v_SegmentationVolumeA, egItkUC3DImagePtr &v_SegmentationVolumeB)
{
    egItkUC3DImagePtr v_newSegmentationVolumeB = casSegmentationEvaluationTools::CreateCommonSpace(v_SegmentationVolumeA, v_SegmentationVolumeB);
    egItkUC3DImagePtr v_newSegmentationVolumeA = casSegmentationEvaluationTools::CreateCommonSpace(v_SegmentationVolumeA, v_SegmentationVolumeB);

    casSegmentationEvaluationTools::PasteIntoCommonSpace(v_newSegmentationVolumeB, v_SegmentationVolumeB);
    casSegmentationEvaluationTools::PasteIntoCommonSpace(v_newSegmentationVolumeA, v_SegmentationVolumeA);

    v_SegmentationVolumeA = v_newSegmentationVolumeA;
    v_SegmentationVolumeB = v_newSegmentationVolumeB;
}

egItkUC3DImagePtr CreateCommonSpace(egItkUC3DImagePtr a_ablationMask, egItkUC3DImagePtr a_tumorMask)
{
    egItkUC3DImagePtr v_newImage;
    v_newImage = egItkUC3DImageType::New();

    egItkUC3DImageType::PointType v_Origin;
    egItkUC3DImageType::PointType v_vectTumor = a_tumorMask->GetOrigin();
    egItkUC3DImageType::PointType v_vectAblation = a_ablationMask->GetOrigin();

    v_Origin[0] = (v_vectTumor.GetElement(0) < v_vectAblation.GetElement(0)) ? v_vectTumor.GetElement(0) : v_vectAblation.GetElement(0);
    v_Origin[1] = (v_vectTumor.GetElement(1) < v_vectAblation.GetElement(1)) ? v_vectTumor.GetElement(1) : v_vectAblation.GetElement(1);
    v_Origin[2] = (v_vectTumor.GetElement(2) < v_vectAblation.GetElement(2)) ? v_vectTumor.GetElement(2) : v_vectAblation.GetElement(2);
    v_newImage->SetOrigin(v_Origin);

    v_newImage->SetSpacing(a_ablationMask->GetSpacing());

    egItkUC3DImageType::SizeType v_sizeTumor = a_tumorMask->GetLargestPossibleRegion().GetSize();

    egItkUC3DImageType::IndexType v_tumorIndex;
    v_tumorIndex[0] = v_sizeTumor.GetElement(0);
    v_tumorIndex[1] = v_sizeTumor.GetElement(1);
    v_tumorIndex[2] = v_sizeTumor.GetElement(2);

    a_tumorMask->TransformIndexToPhysicalPoint(v_tumorIndex, v_vectTumor);

    egItkUC3DImageType::SizeType v_sizeAblation = a_ablationMask->GetLargestPossibleRegion().GetSize();

    egItkUC3DImageType::IndexType v_ablationIndex;
    v_ablationIndex[0] = v_sizeAblation.GetElement(0);
    v_ablationIndex[1] = v_sizeAblation.GetElement(1);
    v_ablationIndex[2] = v_sizeAblation.GetElement(2);

    a_ablationMask->TransformIndexToPhysicalPoint(v_ablationIndex, v_vectAblation);

    egItkUC3DImageType::PointType v_End;
    v_End[0] = (v_vectTumor.GetElement(0) > v_vectAblation.GetElement(0)) ? v_vectTumor.GetElement(0) : v_vectAblation.GetElement(0);
    v_End[1] = (v_vectTumor.GetElement(1) > v_vectAblation.GetElement(1)) ? v_vectTumor.GetElement(1) : v_vectAblation.GetElement(1);
    v_End[2] = (v_vectTumor.GetElement(2) > v_vectAblation.GetElement(2)) ? v_vectTumor.GetElement(2) : v_vectAblation.GetElement(2);

    egItkUC3DImageType::IndexType v_endIndex;
    v_newImage->TransformPhysicalPointToIndex(v_End, v_endIndex);

    egItkUC3DImageType::SizeType v_newSize;
    v_newSize[0] = v_endIndex.GetElement(0);
    v_newSize[1] = v_endIndex.GetElement(1);
    v_newSize[2] = v_endIndex.GetElement(2);

    egItkUC3DImageType::IndexType v_IndexAblation = a_ablationMask->GetLargestPossibleRegion().GetIndex();
    egItkUC3DImageType::IndexType v_IndexTumor = a_tumorMask->GetLargestPossibleRegion().GetIndex();
    egItkUC3DImageType::IndexType v_StartIndex;
    v_StartIndex[0] = (v_IndexTumor.GetElement(0) < v_IndexAblation.GetElement(0)) ? v_IndexTumor.GetElement(0) : v_IndexAblation.GetElement(0);
    v_StartIndex[1] = (v_IndexTumor.GetElement(1) < v_IndexAblation.GetElement(1)) ? v_IndexTumor.GetElement(1) : v_IndexAblation.GetElement(1);
    v_StartIndex[2] = (v_IndexTumor.GetElement(2) < v_IndexAblation.GetElement(2)) ? v_IndexTumor.GetElement(2) : v_IndexAblation.GetElement(2);

    egItkUC3DImageType::IndexType v_ZeroVector;
    v_ZeroVector[0] = 0;
    v_ZeroVector[1] = 0;
    v_ZeroVector[2] = 0;

    if (v_StartIndex != v_ZeroVector)
    {
        v_newImage->TransformIndexToPhysicalPoint(v_StartIndex, v_Origin);
        v_newImage->SetOrigin(v_Origin);
        v_StartIndex = v_ZeroVector;
    }

    egItkUC3DImageType::RegionType v_newRegion;
    v_newRegion.SetSize(v_newSize);
    v_newRegion.SetIndex(v_StartIndex);
    v_newImage->SetRegions(v_newRegion);

    v_newImage->Allocate();

    itk::ImageRegionIteratorWithIndex<egItkUC3DImageType> v_itItkNewImage(v_newImage, v_newImage->GetLargestPossibleRegion());

    v_itItkNewImage.GoToBegin();
    while (!v_itItkNewImage.IsAtEnd()){
        v_itItkNewImage.Set(0);
        ++v_itItkNewImage;
    }

    return v_newImage;
}



egItkUC3DImageType::PointType v_itkPixelLocationPointinSpace;
    egItkUC3DImageType::IndexType v_itkPixelLocationCoordinatesInImage;

    itk::ImageRegionIteratorWithIndex<egItkUC3DImageType> v_itItkNewImage(v_NewBiggerImage, v_NewBiggerImage->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<egItkUC3DImageType> v_itItkOldImage(v_oldImg, v_oldImg->GetLargestPossibleRegion());

    v_itItkOldImage.GoToBegin();
    while (!v_itItkOldImage.IsAtEnd()){
        v_itkPixelLocationCoordinatesInImage = v_itItkOldImage.GetIndex();

        v_oldImg->TransformIndexToPhysicalPoint(v_itkPixelLocationCoordinatesInImage, v_itkPixelLocationPointinSpace);
        v_NewBiggerImage->TransformPhysicalPointToIndex(v_itkPixelLocationPointinSpace, v_itkPixelLocationCoordinatesInImage);
        v_itItkNewImage.SetIndex(v_itkPixelLocationCoordinatesInImage);

        v_itItkNewImage.Set(v_itItkOldImage.Get());

        ++v_itItkOldImage;
    }
