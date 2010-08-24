<?php

namespace Volkszaehler\Model\Proxies;

/**
 * THIS CLASS WAS GENERATED BY THE DOCTRINE ORM. DO NOT EDIT THIS FILE.
 */
class VolkszaehlerModelGroupProxy extends \Volkszaehler\Model\Group implements \Doctrine\ORM\Proxy\Proxy
{
    private $_entityPersister;
    private $_identifier;
    public $__isInitialized__ = false;
    public function __construct($entityPersister, $identifier)
    {
        $this->_entityPersister = $entityPersister;
        $this->_identifier = $identifier;
    }
    private function _load()
    {
        if (!$this->__isInitialized__ && $this->_entityPersister) {
            $this->__isInitialized__ = true;
            if ($this->_entityPersister->load($this->_identifier, $this) === null) {
                throw new \Doctrine\ORM\EntityNotFoundException();
            }
            unset($this->_entityPersister, $this->_identifier);
        }
    }

    
    public function addGroup(\Volkszaehler\Model\Group $child)
    {
        $this->_load();
        return parent::addGroup($child);
    }

    public function removeGroup(\Volkszaehler\Model\Group $child)
    {
        $this->_load();
        return parent::removeGroup($child);
    }

    public function addChannel(\Volkszaehler\Model\Channel $child)
    {
        $this->_load();
        return parent::addChannel($child);
    }

    public function removeChannel(\Volkszaehler\Model\Channel $child)
    {
        $this->_load();
        return parent::removeChannel($child);
    }

    public function getName()
    {
        $this->_load();
        return parent::getName();
    }

    public function setName($name)
    {
        $this->_load();
        return parent::setName($name);
    }

    public function getDescription()
    {
        $this->_load();
        return parent::getDescription();
    }

    public function setDescription($description)
    {
        $this->_load();
        return parent::setDescription($description);
    }

    public function getChildren()
    {
        $this->_load();
        return parent::getChildren();
    }

    public function getParents()
    {
        $this->_load();
        return parent::getParents();
    }

    public function getChannels()
    {
        $this->_load();
        return parent::getChannels();
    }

    public function getInterpreter(\Doctrine\ORM\EntityManager $em)
    {
        $this->_load();
        return parent::getInterpreter($em);
    }

    public function getProperty($name)
    {
        $this->_load();
        return parent::getProperty($name);
    }

    public function getProperties()
    {
        $this->_load();
        return parent::getProperties();
    }

    public function setProperty($name, $value)
    {
        $this->_load();
        return parent::setProperty($name, $value);
    }

    public function unsetProperty($name)
    {
        $this->_load();
        return parent::unsetProperty($name);
    }

    public function getId()
    {
        $this->_load();
        return parent::getId();
    }

    public function getUuid()
    {
        $this->_load();
        return parent::getUuid();
    }


    public function __sleep()
    {
        return array('__isInitialized__', 'id', 'uuid', 'tokens', 'properties', 'channels', 'children', 'parents');
    }
}