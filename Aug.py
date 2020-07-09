import inspect
import numpy as np
import pprint
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList

class Augmentation(metaclass=ABCMeta):
    """
    Augmentation defines policies/strategies to generate :class:`Transform` from data.
    It is often used for pre-processing of input data. A policy typically contains
    randomness, but it can also choose to deterministically generate a :class:`Transform`.

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method with the :attr:`input_args` attribute.
    When called with the positional arguments defined by the :attr:`input_args`,
    the :meth:`get_transform` method executes the policy.

    Examples:
    ::
        # if a policy needs to know both image and semantic segmentation
        assert aug.input_args == ("image", "sem_seg")
        tfm: Transform = aug.get_transform(image, sem_seg)
        new_image = tfm.apply_image(image)

    To implement a custom :class:`Augmentation`, define its :attr:`input_args` and
    implement :meth:`get_transform`.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to apply the actual transform to those data.
    """

    input_args: Tuple[str] = ("image",)
    """
    Attribute of class instances that defines the argument(s) needed by
    :meth:`get_transform`. Default to only "image", because most policies only
    require knowing the image in order to determine the transform.

    Users can freely define arbitrary new args and their types in custom
    :class:`Augmentation`. In detectron2 we use the following convention:

    * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
      floating point in range [0, 1] or [0, 255].
    * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
      of N instances. Each is in XYXY format in unit of absolute coordinates.
    * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

    We do not specify convention for other types and do not include builtin
    :class:`Augmentation` that uses other types in detectron2.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    # NOTE: in the future, can allow it to return list[Augmentation],
    # to delegate augmentation to others

    def get_transform(self, *args) -> Transform:
        """
        Execute the policy to use input data to create transform(s).

        Args:
            arguments must follow what's defined in :attr:`input_args`.

        Returns:
            Return a :class:`Transform` instance, which is the transform to apply to inputs.
        """
        pass


    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()


    __str__ = __repr__



TransformGen = Augmentation

class AugInput:
    """
    A base class for anything on which a list of :class:`Augmentation` can be applied.
    This class provides input arguments for :class:`Augmentation` to use, and defines how
    to apply transforms to these data.

    An instance of this class must satisfy the following:

    * :class:`Augmentation` declares some data it needs as arguments. A :class:`AugInput`
      must provide access to these data in the form of attribute access (``getattr``).
      For example, if a :class:`Augmentation` to be applied needs "image" and "sem_seg"
      arguments, this class must have the attribute "image" and "sem_seg" whose content
      is as required by the :class:`Augmentation`s.
    * This class must have a :meth:`transform(tfm: Transform) -> None` method which
      in-place transforms all attributes stored in the class.
    """

    def transform(self, tfm: Transform) -> None:
        raise NotImplementedError

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Apply a list of Transform/Augmentation in-place and returned the applied transform.
        Attributes of this class will be modified.

        Returns:
            TransformList:
                returns transformed inputs and the list of transforms applied.
                The TransformList can then be applied to other data associated with the inputs.
        """
        tfms = []
        for aug in augmentations:
            if isinstance(aug, Augmentation):
                args = []
                for f in aug.input_args:
                    try:
                        args.append(getattr(self, f))
                    except AttributeError:
                        raise AttributeError(
                            f"Augmentation {aug} needs '{f}', which is not an attribute of {self}!"
                        )

                tfm = aug.get_transform(*args)
                assert isinstance(tfm, Transform), (
                    f"{type(aug)}.get_transform must return an instance of Transform! "
                    "Got {type(tfm)} instead."
                )
            else:
                tfm = aug
            self.transform(tfm)
            tfms.append(tfm)
        return TransformList(tfms)



class StandardAugInput(AugInput):
    """
    A standard implementation of :class:`AugInput` for the majority of use cases.
    This class provides the following standard attributes that are common to use by
    Augmentation (augmentation policies). These are chosen because most
    :class:`Augmentation` won't need anything more to define a augmentation policy.
    After applying augmentations to these special attributes, the returned transforms
    can then be used to transform other data structures that users have.

    Attributes:
        image (ndarray): image in HW or HWC format. The meaning of C is up to users
        boxes (ndarray or None): Nx4 boxes in XYXY_ABS mode
        sem_seg (ndarray or None): HxW semantic segmentation mask

    Examples:
    ::
        input = StandardAugInput(image, boxes=boxes)
        tfms = input.apply_augmentations(list_of_augmentations)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may require augmentation
    policies that need more inputs. An algorithm may need to transform inputs
    in a way different from the standard approach defined in this class. In those
    situations, users can implement new subclasses of :class:`AugInput` with differnt
    attributes and the :meth:`transform` method.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255].
            boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
                of N instances. Each is in XYXY format in unit of absolute coordinates.
            sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.
        """

        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg


    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

class mapper(DatasetMapper):
  def __init__(
        self,
        is_train: bool,
        *,
        augmentations: tfm,
        image_format: 'BGR',
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False
    ):
    super(mapper, self).__init__(cfg)
    self.is_train               = is_train
    self.augmentations          = augmentations
    self.image_format           = image_format
    self.use_instance_mask      = use_instance_mask
    self.instance_mask_format   = instance_mask_format
    self.use_keypoint           = use_keypoint
    self.keypoint_hflip_indices = keypoint_hflip_indices
    self.proposal_topk          = precomputed_proposal_topk
    self.recompute_boxes        = recompute_boxes
    # fmt: on
    logger = logging.getLogger(__name__)
    logger.info("Augmentations used in training: " + str(augmentations))


  def from_config(cls, cfg, is_train: bool = True):
        augs = tfm
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

  def __call__(self, dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = utils.read_image(dataset_dict["file_name"], format='BGR')
    utils.check_image_size(dataset_dict, image)
    # USER: Remove if you don't do semantic/panoptic segmentation.
    if "sem_seg_file_name" in dataset_dict:
        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
    else:
        sem_seg_gt = None
    aug_input = StandardAugInput(image, sem_seg=sem_seg_gt)
    transforms = aug_input.apply_augmentations(self.augmentations)
    image, sem_seg_gt = aug_input.image, aug_input.sem_seg
    image_shape = image.shape[:2]  # h, w
    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if sem_seg_gt is not None:
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
    # USER: Remove if you don't use pre-computed proposals.
    # Most users would not need this feature.
    if self.proposal_topk is not None:
        utils.transform_proposals(
            dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
        )
    if not self.is_train:
        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict
    if "annotations" in dataset_dict:
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)
        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict